"""
The ORACLE: benchmark a real compiled kernel config and measure it.

The answer here is right or wrong by EXECUTION, not opinion:
  - correctness = every config produces the same matmul (checksums agree), and
  - cost        = measured wall-clock time of the compiled binary (best-of-k).

`brute_force_best` is the EXPENSIVE search we train the net to replace: it runs
every one of the 16 compiled configs and returns the winner. That is the cost we
want to avoid at inference time -- the net should jump straight to a near-optimal
config, and the oracle merely VERIFIES it (one extra run) when we want a
no-regression guarantee.
"""
import numpy as np

import codegen


def profile_all(M, N, K, seed=0):
    """Time every config on a shape. Returns (times_array, checksums) aligned
    to codegen.CONFIGS, times in milliseconds."""
    times = np.empty(codegen.N_CONFIGS, np.float32)
    sums = np.empty(codegen.N_CONFIGS, np.float64)
    for i, cfg in enumerate(codegen.CONFIGS):
        ms, chk = codegen.run(cfg, M, N, K, seed)
        times[i] = ms
        sums[i] = chk
    return times, sums


def brute_force_best(M, N, K, seed=0):
    """The search we replace: benchmark all configs, return (best_idx, times)."""
    times, _ = profile_all(M, N, K, seed)
    return int(times.argmin()), times
