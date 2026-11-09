"""Compute the masking numbers: expected masked fraction and mean span length.

Run: python3 scripts/masking_example.py
Section 3.1 and Section 4.2: p = 0.065, M = 10; the paper reports about 49% of
time steps masked and a mean span length of 14.7 (Appendix A).
"""

import numpy as np

p, M = 0.065, 10
print("1 - (1-p)^M             =", round(1 - (1 - p) ** M, 4))

T = 781  # frames in a 15.6 s BASE crop
rng = np.random.default_rng(0)
fracs, lens = [], []
for _ in range(2000):
    n = round(p * T)
    starts = rng.choice(T, size=n, replace=False)
    mask = np.zeros(T, bool)
    for s in starts:
        mask[s:s + M] = True
    fracs.append(mask.mean())
    padded = np.concatenate([[0], mask.astype(int), [0]])
    edges = np.flatnonzero(np.diff(padded))
    lens.extend((edges[1::2] - edges[0::2]).tolist())
print("T =", T, "simulated masked fraction =", round(float(np.mean(fracs)), 4))
print("simulated mean span length         =", round(float(np.mean(lens)), 2))
print("span length in ms (x20 ms)         =", round(float(np.mean(lens)) * 20))

# One small hand-checkable case.
T2, starts, M2 = 12, [1, 4, 6], 3
mask = np.zeros(T2, bool)
for s in starts:
    mask[s:s + M2] = True
print("T=12 starts [1,4,6] M=3 mask       =", mask.astype(int).tolist(), "count", int(mask.sum()))
