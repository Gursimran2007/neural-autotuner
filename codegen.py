"""
The SUBSTRATE: generate, compile, and run real C matmul kernels.

Why C and not NumPy: NumPy's `@` calls BLAS, which has already solved cache
tiling internally -- so a NumPy "tile knob" is near-trivial and the optimum
barely moves with shape. There is no rich landscape to learn. By emitting our
OWN C kernel with three interacting knobs we restore a genuine, shape-dependent,
non-monotonic cost landscape (measured spread best/worst ~7-11x), which is the
thing the autotuner exists to navigate.

Knobs (a "config"):
  TILE   in {32,64,128,256}  -- cache-block size
  UNROLL in {1,4,8}          -- inner-loop unroll factor (ikj order only)
  ORDER  in {0,1}            -- 0 = ikj streaming (unrollable),
                                1 = ijk accumulate-in-register (UNROLL irrelevant)

Because ORDER=1 ignores UNROLL, those configs collapse to one per TILE -- so the
real config space is deduped to ~16 unique kernels. Each unique config is
compiled ONCE into its own cached binary; sweeping thousands of shapes is then
just running the precompiled binaries (fast), never recompiling.

A binary, given `M N K [SEED]` on argv, fills A,B from a seeded pattern, runs the
kernel best-of-3, and prints two numbers: best milliseconds and a checksum of C.
The checksum lets the oracle VERIFY every config computes the same matmul (all
configs must agree to ~rel-tol) -- correctness by execution, not by trust.
"""
import hashlib
import os
import subprocess
import time

# ---- the config space -------------------------------------------------------
TILES = [32, 64, 128, 256]
UNROLLS = [1, 4, 8]
ORDERS = [0, 1]


def all_configs():
    """Deduped list of (TILE, UNROLL, ORDER). ORDER=1 ignores UNROLL, so we
    keep only UNROLL=1 there to avoid compiling identical binaries."""
    cfgs = []
    for o in ORDERS:
        for t in TILES:
            if o == 1:
                cfgs.append((t, 1, o))
            else:
                for u in UNROLLS:
                    cfgs.append((t, u, o))
    return cfgs


CONFIGS = all_configs()
N_CONFIGS = len(CONFIGS)

# ---- the C source (multi-knob, self-checking) -------------------------------
C_SRC = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef TILE
#define TILE 64
#endif
#ifndef UNROLL
#define UNROLL 1
#endif
#ifndef ORDER
#define ORDER 0
#endif
void mm(const float* restrict A,const float* restrict B,float* restrict C,int M,int N,int K){
  for(long i=0;i<(long)M*N;i++)C[i]=0;
  for(int i0=0;i0<M;i0+=TILE)for(int k0=0;k0<K;k0+=TILE)for(int j0=0;j0<N;j0+=TILE){
    int iM=i0+TILE<M?i0+TILE:M,kM=k0+TILE<K?k0+TILE:K,jM=j0+TILE<N?j0+TILE:N;
#if ORDER==0
    for(int i=i0;i<iM;i++)for(int k=k0;k<kM;k++){float a=A[i*K+k];
      int j=j0; for(;j+UNROLL<=jM;j+=UNROLL){
        for(int u=0;u<UNROLL;u++)C[i*N+j+u]+=a*B[k*N+j+u];}
      for(;j<jM;j++)C[i*N+j]+=a*B[k*N+j];}
#else
    for(int i=i0;i<iM;i++)for(int j=j0;j<jM;j++){float s=C[i*N+j];
      for(int k=k0;k<kM;k++)s+=A[i*K+k]*B[k*N+j]; C[i*N+j]=s;}
#endif
  }
}
int main(int c,char**v){
  int M=atoi(v[1]),N=atoi(v[2]),K=atoi(v[3]);
  int SEED=c>4?atoi(v[4]):0;
  float*A=malloc((long)M*K*4),*B=malloc((long)K*N*4),*C=malloc((long)M*N*4);
  for(long i=0;i<(long)M*K;i++)A[i]=(float)(((i*1103515245L+12345L+SEED)%1000))*0.001f;
  for(long i=0;i<(long)K*N;i++)B[i]=(float)(((i*12345L+1103515245L+SEED)%1000))*0.001f;
  mm(A,B,C,M,N,K);
  struct timespec s,e;double best=1e9;
  for(int r=0;r<3;r++){clock_gettime(CLOCK_MONOTONIC,&s);
    mm(A,B,C,M,N,K);clock_gettime(CLOCK_MONOTONIC,&e);
    double ms=(e.tv_sec-s.tv_sec)*1e3+(e.tv_nsec-s.tv_nsec)/1e6;if(ms<best)best=ms;}
  double sum=0; for(long i=0;i<(long)M*N;i++)sum+=C[i];
  printf("%.4f %.6e\n",best,sum);
  return 0;
}
"""

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".kbin")
CC = os.environ.get("CC", "cc")
CFLAGS = ["-O2", "-march=native"]


def _src_path():
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, "kernel.c")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(C_SRC)
    return p


def _bin_path(cfg):
    t, u, o = cfg
    return os.path.join(CACHE_DIR, f"mm_T{t}_U{u}_O{o}")


def compile_config(cfg):
    """Compile a config into a cached binary (no-op if already built)."""
    t, u, o = cfg
    out = _bin_path(cfg)
    if os.path.exists(out):
        return out
    src = _src_path()
    cmd = [CC, *CFLAGS,
           f"-DTILE={t}", f"-DUNROLL={u}", f"-DORDER={o}",
           src, "-o", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def compile_all(verbose=True):
    """Precompile every unique config once. Returns list of binary paths."""
    paths = []
    for i, cfg in enumerate(CONFIGS):
        paths.append(compile_config(cfg))
        if verbose:
            print(f"  compiled {i+1}/{N_CONFIGS}  config={cfg}")
    return paths


def run(cfg, M, N, K, seed=0):
    """Run a config's binary on a shape. Returns (best_ms, checksum)."""
    out = compile_config(cfg)
    r = subprocess.run([out, str(M), str(N), str(K), str(seed)],
                       check=True, capture_output=True, text=True)
    ms_str, sum_str = r.stdout.split()
    return float(ms_str), float(sum_str)


def verify_configs(M=128, N=128, K=128, seed=1, rtol=1e-3):
    """Sanity check: every config must compute the same matmul (checksums
    agree to rtol). Returns (ok, checksums). Correctness by execution."""
    sums = []
    for cfg in CONFIGS:
        _, s = run(cfg, M, N, K, seed)
        sums.append(s)
    ref = sums[0]
    ok = all(abs(s - ref) <= rtol * max(1.0, abs(ref)) for s in sums)
    return ok, sums


if __name__ == "__main__":
    print(f"config space: {N_CONFIGS} unique kernels")
    for c in CONFIGS:
        print("   ", c)
    print("\ncompiling all configs ...")
    t0 = time.perf_counter()
    compile_all()
    print(f"compiled in {time.perf_counter()-t0:.1f}s\n")
    print("verifying all configs agree on the result ...")
    ok, sums = verify_configs()
    print(f"  agreement: {'OK' if ok else 'MISMATCH'}  "
          f"(checksum spread {max(sums)-min(sums):.2e})")
    # quick landscape peek on a few shapes
    print("\nlandscape sanity (best/worst time spread per shape):")
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(5):
        M, N, K = (int(rng.integers(2, 17)) * 32 for _ in range(3))
        ts = [run(c, M, N, K)[0] for c in CONFIGS]
        ts = np.array(ts)
        bi = int(ts.argmin())
        print(f"  {M:4d}x{N:4d}x{K:4d}  spread {ts.max()/ts.min():5.1f}x  "
              f"best={CONFIGS[bi]}  ({ts.min():.3f} ms)")
