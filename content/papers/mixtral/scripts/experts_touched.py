"""Compute how many experts a batch touches in one layer, with uniform random routing.

Run: python3 scripts/experts_touched.py

With n = 8 experts and top-2 routing, one token leaves a given expert unused
with probability 1 - 2/8 = 3/4. For a batch of B independent tokens, an expert
is untouched with probability (3/4)^B.
"""

import numpy as np

N, K = 8, 2
EXPERT_PARAMS = 3 * 4096 * 14336  # one SwiGLU expert, Table 1
rng = np.random.default_rng(0)

print("B   expected experts touched   simulated   expert weights read / active weights")
for B in [1, 2, 4, 8, 16, 32]:
    expected = N * (1 - (1 - K / N) ** B)
    sims = [len({e for _ in range(B) for e in rng.permutation(N)[:K]}) for _ in range(4000)]
    ratio = expected / K
    print(f"{B:<3} {expected:>25.3f} {np.mean(sims):>11.3f} {ratio:>20.2f}x")
print("bytes of expert weights per layer in bfloat16, per expert =", 2 * EXPERT_PARAMS)
