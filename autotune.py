"""
The PRODUCT: pick a near-optimal kernel config for a shape in one shot.

    /opt/anaconda3/bin/python autotune.py 512 512 512

Trains (or reuses) the cost-model net, predicts the config for the requested
M,N,K with ONE forward pass -- no brute-force search -- then VERIFIES it by
actually running it and comparing against the safe fixed-config fallback. Prints
the chosen config, its measured time, and how it compares to the brute-force
optimum (so the claim is checkable, never trusted).
"""
import pickle
import sys

import numpy as np

import codegen
import data as D
import oracle
from mlp import MLP

CKPT = "autotuner.pkl"


def train_and_save(samples_path="samples.pkl"):
    import math
    samples = D.load(samples_path)
    X = np.stack([s["x"] for s in samples])
    L = np.log(np.stack([s["times"] for s in samples]))
    Y = (L - L.mean(axis=1, keepdims=True)).astype(np.float32)
    net = MLP(X.shape[1], codegen.N_CONFIGS, hidden=(96, 96), seed=0)
    m, v = net.zeros_like_params(), net.zeros_like_params()
    lr0, epochs, bs, t = 0.01, 500, 64, 0
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        lr = lr0 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / epochs))
        for k in range(0, len(X), bs):
            bi = idx[k:k + bs]
            t += 1
            P = net.forward(X[bi], cache=True)
            diff = P - Y[bi]
            dcur = (2.0 / diff.size) * diff
            gW, gb = [None] * len(net.W), [None] * len(net.b)
            for li in reversed(range(len(net.W))):
                gW[li] = net._acts[li].T @ dcur + 1e-4 * net.W[li]
                gb[li] = dcur.sum(axis=0, keepdims=True)
                if li > 0:
                    dcur = (dcur @ net.W[li].T) * (net._pres[li - 1] > 0)
            net.step((gW, gb), lr, m, v, t)
    train_mean_r = [np.mean([s["times"][ci] / s["times"].min() for s in samples])
                    for ci in range(codegen.N_CONFIGS)]
    fixed_idx = int(np.argmin(train_mean_r))
    with open(CKPT, "wb") as f:
        pickle.dump({"W": net.W, "b": net.b, "fixed_idx": fixed_idx}, f)
    return net, fixed_idx


def load_or_train():
    import os
    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            d = pickle.load(f)
        net = MLP(7, codegen.N_CONFIGS, hidden=(96, 96), seed=0)
        net.W, net.b = d["W"], d["b"]
        return net, d["fixed_idx"]
    print("no checkpoint; training from samples.pkl ...")
    return train_and_save()


def main():
    if len(sys.argv) < 4:
        print("usage: autotune.py M N K"); return
    M, N, K = (int(x) for x in sys.argv[1:4])
    net, fixed_idx = load_or_train()

    x = D.features(M, N, K)[None, :]
    pred_idx = int(net.forward(x).argmin(axis=1)[0])
    pred_cfg = codegen.CONFIGS[pred_idx]
    fixed_cfg = codegen.CONFIGS[fixed_idx]

    # VERIFY by execution: time the prediction and the safe fallback, keep faster
    t_pred, _ = codegen.run(pred_cfg, M, N, K)
    t_fixed, _ = codegen.run(fixed_cfg, M, N, K)
    if t_pred <= t_fixed:
        chosen_cfg, t_chosen, why = pred_cfg, t_pred, "net prediction"
    else:
        chosen_cfg, t_chosen, why = fixed_cfg, t_fixed, "fell back to safe fixed config"

    # ground truth (only for the report -- a product wouldn't pay this)
    best_idx, times = oracle.brute_force_best(M, N, K)
    best_cfg = codegen.CONFIGS[best_idx]
    t_opt = times.min()

    print(f"\nshape {M}x{N}x{K}")
    print(f"  net predicted config : {pred_cfg}   ({t_pred:.3f} ms)")
    print(f"  safe fixed config    : {fixed_cfg}   ({t_fixed:.3f} ms)")
    print(f"  CHOSEN (verified)    : {chosen_cfg}   ({t_chosen:.3f} ms)  [{why}]")
    print(f"  brute-force optimum  : {best_cfg}   ({t_opt:.3f} ms)  "
          f"[cost: ran all {codegen.N_CONFIGS} configs]")
    print(f"  regret of our choice : {t_chosen / t_opt:.3f}x   "
          f"(1.000 = matched the optimum; we ran 2 configs, not {codegen.N_CONFIGS})")


if __name__ == "__main__":
    main()
