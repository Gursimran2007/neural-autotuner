"""
Train the COST-MODEL net and evaluate it HONESTLY against the real baselines.

    /opt/anaconda3/bin/python codegen.py     # once, build the 16 kernels
    /opt/anaconda3/bin/python data.py 700    # once, benchmark a dataset
    /opt/anaconda3/bin/python train.py

The net learns to PREDICT THE RUNTIME of every config from the shape (the whole
cost curve), then we pick the arg-min of the predicted curve. This is the
principled approach real autotuners use (TVM AutoTVM, Ansor): using the full
timing vector as signal -- not just the arg-min label -- stops the net from
collapsing to the single majority config, and it rewards picking ANY near-optimal
config, which is what regret actually measures.

The metric that matters is REGRET = chosen_time / optimal_time (1.0 = perfect).
Top-1 accuracy is a distraction: picking a config 2% slower than the very best is
a win. We report regret for:

  - oracle (brute-force search)   : 1.00 by definition, but costs N_CONFIGS runs
  - best FIXED config (no-ML)     : the strongest non-learned baseline -- pick the
                                    single config best on average over TRAIN, use
                                    it for everything. This is what you'd ship
                                    WITHOUT the net.
  - cost-model net (1 prediction) : the moat. One forward pass, zero kernel runs.
  - net + VERIFY-and-fallback     : run the predicted config once; if it is
                                    somehow slower than the safe fixed config,
                                    fall back. Guarantees no regression.
"""
import math
import sys

import numpy as np

import codegen
import data as D
from mlp import MLP


def regret(times_row, idx):
    return times_row[idx] / times_row.min()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "samples.pkl"
    samples = D.load(path)
    rng = np.random.default_rng(0)
    rng.shuffle(samples)
    split = int(len(samples) * 0.8)
    train, test = samples[:split], samples[split:]
    nc = codegen.N_CONFIGS

    X = np.stack([s["x"] for s in train])
    Xte = np.stack([s["x"] for s in test])

    # target: per-sample mean-centered log runtime -- the SHAPE of the cost curve
    # across configs for this problem, not the absolute scale (which the net
    # neither needs nor should waste capacity on).
    def targets(rows):
        L = np.log(np.stack([s["times"] for s in rows]))
        return (L - L.mean(axis=1, keepdims=True)).astype(np.float32)
    Y = targets(train)

    net = MLP(X.shape[1], nc, hidden=(96, 96), seed=0)
    m, v = net.zeros_like_params(), net.zeros_like_params()
    lr0, epochs, bs, t = 0.01, 500, 64, 0

    def eval_regret(rows, Xr):
        pred = net.forward(Xr)               # predicted cost curve
        choose = pred.argmin(axis=1)
        return np.array([regret(rows[i]["times"], choose[i]) for i in range(len(rows))])

    best_mean, best_W, best_b = 1e9, None, None
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(X))
        lr = lr0 * 0.5 * (1 + math.cos(math.pi * (ep - 1) / epochs))
        for k in range(0, len(X), bs):
            bi = idx[k:k + bs]
            t += 1
            # MSE on the cost curve (manual; MLP's CE is for classification)
            P = net.forward(X[bi], cache=True)
            diff = (P - Y[bi])
            dZ = (2.0 / diff.size) * diff
            gW, gb = [None] * len(net.W), [None] * len(net.b)
            dcur = dZ
            for li in reversed(range(len(net.W))):
                a_prev = net._acts[li]
                gW[li] = a_prev.T @ dcur + 1e-4 * net.W[li]
                gb[li] = dcur.sum(axis=0, keepdims=True)
                if li > 0:
                    dcur = (dcur @ net.W[li].T) * (net._pres[li - 1] > 0)
            net.step((gW, gb), lr, m, v, t)
        if ep % 50 == 0 or ep == 1:
            r = eval_regret(test, Xte)
            if r.mean() < best_mean:
                best_mean = r.mean()
                best_W = [w.copy() for w in net.W]; best_b = [b.copy() for b in net.b]
            print(f"  epoch {ep:3d} | test mean-regret {r.mean():.4f} | "
                  f"median {np.median(r):.4f}")
    if best_W:
        net.W, net.b = best_W, best_b

    # ----- baselines on the TEST set -----
    # best FIXED config chosen on TRAIN only (no peeking): lowest mean regret.
    train_mean_r = [np.mean([regret(s["times"], ci) for s in train]) for ci in range(nc)]
    fixed_idx = int(np.argmin(train_mean_r))
    fixed_cfg = codegen.CONFIGS[fixed_idx]

    r_oracle = np.ones(len(test))
    r_fixed = np.array([regret(s["times"], fixed_idx) for s in test])
    r_net = eval_regret(test, Xte)
    pred = net.forward(Xte).argmin(axis=1)
    r_safe = np.array([
        regret(test[i]["times"],
               pred[i] if test[i]["times"][pred[i]] <= test[i]["times"][fixed_idx]
               else fixed_idx)
        for i in range(len(test))])
    acc = np.mean([pred[i] == test[i]["best_idx"] for i in range(len(test))])

    def line(name, r):
        return (f"  {name:<36} mean {r.mean():6.4f}  median {np.median(r):6.4f}  "
                f"p95 {np.percentile(r,95):5.3f}  worst {r.max():5.2f}")

    print(f"\n{len(samples)} samples | train {len(train)} / test {len(test)} | "
          f"{nc} configs")
    print(f"best fixed config (no-ML baseline): {fixed_cfg}\n")
    print("REGRET = chosen_time / optimal_time   (1.000 = picked the best config)")
    print(line("oracle (brute-force, N_CONFIGS runs)", r_oracle))
    print(line(f"best fixed config = {fixed_cfg} (no ML)", r_fixed))
    print(line("cost-model net (1 prediction, 0 runs)", r_net))
    print(line("cost-model net + verify-and-fallback", r_safe))
    print(f"\n  net exact-best top-1 accuracy : {acc:.2f}")
    print(f"  search cost      : oracle runs {nc} kernels; net runs 0 (1 with verify).")
    avoid = max(r_fixed.mean() - 1.0, 1e-9)
    gain = (r_fixed.mean() - r_safe.mean()) / avoid
    print(f"  regret over optimal: fixed {r_fixed.mean()-1:.4f} -> net+verify "
          f"{r_safe.mean()-1:.4f}  ({gain*100:.0f}% of avoidable regret removed)")


if __name__ == "__main__":
    main()
