# neural-autotuner

Pick a near-optimal matmul kernel config for a given problem shape **in one
forward pass**, instead of brute-force benchmarking every config. A from-scratch
NumPy cost-model net (no PyTorch/TF) learns the per-config runtime curve from a
self-generated, exactly-labelled benchmark dataset, then a cheap **sound
verifier** (run it once, fall back if it's slower than the safe default)
guarantees the result is never worse than the no-ML baseline.

This is the same shape of moat as a verified decompiler: an **objective oracle**
(measured wall-clock time) + **infinite free labels** (just run the kernel) + an
**own trained net** that turns an expensive search into a cheap prediction.

## The substrate (and an honest detour)

The first version tiled matmul in NumPy. That was a dead end: NumPy's `@` calls
BLAS, which has *already* solved cache tiling internally, so the "tile knob" was
near-trivial and the optimum barely moved with shape — there was no rich
landscape to learn. (Recorded honestly rather than faked.)

So the kernel is now **real C we emit ourselves** (`codegen.py`) with three
interacting knobs:

| knob   | values            | effect                                        |
|--------|-------------------|-----------------------------------------------|
| TILE   | 32, 64, 128, 256  | cache-block size                              |
| UNROLL | 1, 4, 8           | inner-loop unroll (ikj order only)            |
| ORDER  | 0 (ikj) / 1 (ijk) | streaming vs accumulate-in-register           |

ORDER=1 ignores UNROLL, so the space dedupes to **16 unique kernels**, each
compiled **once** into a cached binary. This restores a genuine, shape-dependent,
non-monotonic landscape: **median best/worst spread ≈ 9.8×**, and the winning
config really does change with shape.

Correctness is **by execution**: every config prints a checksum of its result and
they must all agree (they do, to 0.00e+00) — so we know all 16 kernels compute the
same matmul before we ever trust a timing.

## Results (700 self-benchmarked shapes, 80/20 split, this machine)

`REGRET = chosen_time / optimal_time` (1.000 = picked the very best config):

| strategy                        | mean regret | p95   | worst | kernel runs |
|---------------------------------|-------------|-------|-------|-------------|
| oracle (brute force)            | 1.000       | 1.000 | 1.00  | 16          |
| best fixed config (no ML)       | 1.039       | 1.234 | 1.41  | 0           |
| cost-model net (1 prediction)   | 1.033       | 1.088 | **3.54** | 0        |
| **net + verify-and-fallback**   | **1.015**   | 1.082 | 1.41  | 1           |

**Honest reading:**
- The no-ML baseline is already decent (3.9% off optimal) because this landscape
  has a dominant config. That is the bar to beat, and we report it up front.
- The net **alone** averages slightly better but has a **worst case of 3.54×** — a
  single prediction can occasionally be disastrous. This is the whole reason the
  verifier exists.
- **net + verify** removes **62% of the avoidable regret** (3.9% → 1.5% over
  optimal), is provably **never worse** than the fixed baseline (worst = 1.41×,
  identical), and costs **1 kernel run instead of 16** — a 16× cheaper search with
  a no-regression guarantee.

Numbers are machine-specific; re-run to reproduce on your hardware.

## Files

| file          | role                                                              |
|---------------|-------------------------------------------------------------------|
| `codegen.py`  | emit/compile-cache the 16 C kernels; run + verify (checksums)     |
| `oracle.py`   | benchmark a config; brute-force best (the search we replace)      |
| `data.py`     | self-supervised dataset: shape → per-config runtimes (free labels)|
| `mlp.py`      | from-scratch NumPy MLP (He init, ReLU, Adam, manual backprop)     |
| `train.py`    | train the cost-model net + the honest regret eval above           |
| `autotune.py` | the product: predict a config for a shape, then verify by running |

## Run

```sh
python codegen.py        # build + verify the 16 kernels, peek the landscape
python data.py 700       # benchmark a dataset (free exact labels)
python train.py          # train cost-model net + honest regret report
python autotune.py 512 512 512   # one-shot config pick, verified against the optimum
```

Requires only `numpy` and a C compiler (`cc`, with `-march=native`).
