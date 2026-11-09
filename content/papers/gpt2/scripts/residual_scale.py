"""Residual-path variance with and without the 1/sqrt(N) initialization (Section 2.3).

Each residual layer adds a branch output with unit variance. The paper scales
the weights of residual layers by 1/sqrt(N), so a branch's variance drops by N.

Also checks the pre-LN identity property: with all branch weights at zero, a
pre-LN block returns its input, while a post-LN block normalizes it.

Run: python3 scripts/residual_scale.py
"""

import numpy as np

rng = np.random.default_rng(0)
dim, samples = 4096, 1
for blocks in (12, 48):
    N = 2 * blocks  # one attention and one feed-forward residual layer per block
    x = rng.standard_normal(dim)
    plain = x.copy()
    scaled = x.copy()
    for _ in range(N):
        branch = rng.standard_normal(dim)  # stand-in for a unit-variance branch output
        plain += branch
        scaled += branch / np.sqrt(N)
    print(f"{blocks:>2} blocks, N = {N:>2}: variance without scaling = {plain.var():6.2f} (theory {1 + N}), "
          f"with 1/sqrt(N) = {scaled.var():.2f} (theory {1 + 1})")


def layer_norm(v):
    return (v - v.mean()) / v.std()


x = np.array([100.0, 0.0, 0.0, 0.0])
zero_branch = lambda v: 0 * v
print("\ninput          =", x.tolist())
print("pre-LN block   =", (x + zero_branch(layer_norm(x))).tolist())
print("post-LN block  =", np.round(layer_norm(x + zero_branch(x)), 4).tolist())
