#!/usr/bin/env python3
"""pUCT with min-max normalized Q (Appendix B, Equations 2 and 5)."""

import math

import numpy as np

from _load import load

m = load("1-pucb-select")

Q = np.array([2.0, 5.0, 0.0])
N = np.array([10.0, 2.0, 0.0])
P = np.array([0.6, 0.3, 0.1])
lo, hi = 0.0, 10.0
total = N.sum()

print("== One node, tree range [0, 10]")
print("normalized Q:", np.round(m.minmax_normalize(Q, lo, hi), 4))
factor = 1.25 + math.log((total + 19652 + 1) / 19652)
print(f"sum N = {total:.0f}, sqrt = {math.sqrt(total):.4f}, factor c1 + log(...) = {factor:.6f}")
bonus = P * math.sqrt(total) / (1 + N) * factor
print("bonus:", np.round(bonus, 4))
scores = m.puct_scores(Q, N, P, lo, hi)
print("score:", np.round(scores, 4), "-> picks", m.select_action(Q, N, P, lo, hi))

print("\n== The factor c1 + log((sum N + c2 + 1) / c2) for c1 = 1.25, c2 = 19652")
for s in [0, 1, 50, 800, 19652, 100000, 1000000]:
    print(f"sum N = {s:>8}: {1.25 + math.log((s + 19652 + 1) / 19652):.4f}")

print("\n== The same node when Q is on the reward scale of Atari (x 1000)")
raw = m.puct_scores(Q * 1000, N, P, 0.0, 10000.0)
print("normalized scores are unchanged:", np.round(raw, 4))
unnormalized = Q * 1000 + P * math.sqrt(total) / (1 + N) * factor
print("without normalization:", np.round(unnormalized, 1), "-> picks", int(np.argmax(unnormalized)))
