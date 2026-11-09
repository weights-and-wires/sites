#!/usr/bin/env python3
"""Fit a linear reward to synthetic preferences (Section 2.2.3) and sweep the query count."""

import numpy as np
from pref_lib import fit_reward, make_pairs, seg_sum

D, K = 6, 25
w_true = np.array([1.0, -2.0, 0.5, 0.0, 1.5, -1.0])
SEEDS = 20


def run(n, noise):
    cors, accs = [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        F1, F2, mu = make_pairs(rng, n, K, D, w_true, noise)
        w = fit_reward(F1, F2, mu, lr=0.5, steps=400)
        T1, T2, _ = make_pairs(np.random.default_rng(1000 + seed), 2000, K, D, w_true)
        steps = np.concatenate([T1.reshape(-1, D), T2.reshape(-1, D)])
        cors.append(np.corrcoef(steps @ w, steps @ w_true)[0, 1])
        pred = seg_sum(T1, w) > seg_sum(T2, w)
        true = seg_sum(T1, w_true) > seg_sum(T2, w_true)
        accs.append((pred == true).mean())
    return np.mean(cors), np.mean(accs)


print(f"d={D} features, k={K} steps per segment, mean of {SEEDS} seeds")
print("queries | clean labels: corr, pair acc | 10% random labels: corr, pair acc")
for n in [10, 25, 50, 100, 200, 400, 800]:
    c0, a0 = run(n, 0.0)
    c1, a1 = run(n, 0.1)
    print(f"{n:7d} | {c0:.3f} {a0:.3f}            | {c1:.3f} {a1:.3f}")
