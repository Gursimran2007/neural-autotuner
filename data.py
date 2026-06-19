"""
Self-supervised data engine: generate (matrix shape -> per-config runtimes).

Labels are FREE and EXACT -- we just run the compiled kernels and time them. No
annotation, no scraping. This is half the moat: an infinite, perfectly-labelled
dataset that no LLM can obtain (it is YOUR machine's measured cost curve). Each
sample is:

    features(M,N,K)  ->  measured ms for every config in codegen.CONFIGS

from which we derive the best config and, crucially, the REGRET of any config
(how much slower than optimal it is) -- the quantity the autotuner minimises.
"""
import math
import pickle

import numpy as np

import codegen
import oracle


def features(M, N, K):
    """Shape -> a small, scale-aware feature vector for the predictor.

    Log dimensions (cache behaviour is roughly log-scaled), anisotropy ratios,
    and the arithmetic-intensity proxy MNK/(MN+NK+MK) which captures how
    compute- vs memory-bound the op is -- the thing that moves the optimum.
    """
    lm, ln, lk = math.log2(M), math.log2(N), math.log2(K)
    flops = M * N * K
    bytes_touched = M * N + N * K + M * K
    intensity = flops / bytes_touched
    return np.array([
        lm / 9.0, ln / 9.0, lk / 9.0,           # log dims, ~[0,1] for <=512
        lm - ln, ln - lk,                        # anisotropy
        math.log2(intensity) / 9.0,              # arithmetic intensity
        math.log2(flops) / 27.0,                 # total work
    ], dtype=np.float32)


def random_shape(rng):
    """A realistic-ish matmul shape: each dim a multiple of 32 in [64, 512]."""
    pick = lambda: int(rng.integers(2, 17)) * 32   # 64..512
    return pick(), pick(), pick()


def generate(n, seed=0, log_every=25):
    """Generate n samples. Returns list of dicts with features, times, best."""
    codegen.compile_all(verbose=False)   # ensure all 16 binaries exist
    rng = np.random.default_rng(seed)
    data = []
    for i in range(n):
        M, N, K = random_shape(rng)
        times, _ = oracle.profile_all(M, N, K, seed=i)
        best_idx = int(times.argmin())
        data.append({
            "MNK": (M, N, K),
            "x": features(M, N, K),
            "times": times,                # ms per config, aligned to CONFIGS
            "best_idx": best_idx,
            "best_cfg": codegen.CONFIGS[best_idx],
        })
        if log_every and (i + 1) % log_every == 0:
            print(f"  generated {i+1}/{n}")
    return data


def save(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    print(f"generating {n} benchmarked samples over {codegen.N_CONFIGS} configs ...")
    d = generate(n)
    save(d, "samples.pkl")
    spreads = [s["times"].max() / s["times"].min() for s in d]
    from collections import Counter
    best_counts = Counter(s["best_cfg"] for s in d)
    print(f"saved samples.pkl ({len(d)} samples)")
    print(f"median best/worst speed spread: {np.median(spreads):.1f}x  "
          f"(if ~1x, the knob wouldn't matter)")
    print("best-config distribution:")
    for cfg, c in sorted(best_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cfg}: {c}")
