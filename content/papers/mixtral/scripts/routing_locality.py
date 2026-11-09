"""Compute the random-assignment baselines for Table 5 and check them by simulation.

Run: python3 scripts/routing_locality.py
"""

import numpy as np

N, K = 8, 2
first = 1 / N
either = 1 - (6 / 8) * (5 / 7)
print("expected repeat, first choice          =", f"{first:.4f}")
print("expected repeat, first or second choice =", f"{either:.4f}", "(paper: about 46%)")

rng = np.random.default_rng(0)
T = 200_000
idx = np.array([rng.permutation(N)[:K] for _ in range(T)])
same_first = np.mean(idx[1:, 0] == idx[:-1, 0])
overlap = np.mean([len(set(a) & set(b)) > 0 for a, b in zip(idx[1:], idx[:-1])])
print("simulated, first choice                =", f"{same_first:.4f}")
print("simulated, share any expert            =", f"{overlap:.4f}")

# A sticky router: with probability 0.5 a token repeats the previous pair.
sticky = [idx[0]]
for t in range(1, T):
    sticky.append(sticky[-1] if rng.random() < 0.5 else idx[t])
sticky = np.array(sticky)
print("sticky router (p=0.5), first choice    =", f"{np.mean(sticky[1:, 0] == sticky[:-1, 0]):.4f}")
