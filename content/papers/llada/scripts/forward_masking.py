"""Compute the numbers for the forward masking example.

Run: python3 scripts/forward_masking.py
"""

import numpy as np

L = 8
MASK = -1  # stands in for the mask token M

# One sequence at t = 0.5 (Section 2.1): every token is masked independently with probability t.
rng = np.random.default_rng(0)
x0 = np.array([5, 3, 9, 1, 7, 2, 8, 4])
t = 0.5
masked = rng.random(L) < t
xt = np.where(masked, MASK, x0)
print("x0     =", x0.tolist())
print("masked =", masked.astype(int).tolist())
print("x_t    =", xt.tolist(), "(-1 marks M)")

# Mask count is Binomial(L, t), so the mean is L * t.
n = 200_000
for t in (0.25, 0.5, 0.75, 1.0):
    m = rng.random((n, L)) < t
    print(f"t={t:4.2f}  expected masks L*t={L * t:4.2f}  empirical mean={m.sum(axis=1).mean():.4f}"
          f"  P(no mask)={(1 - t) ** L:.4f}  empirical={(m.sum(axis=1) == 0).mean():.4f}")

# SFT masking (Section 2.3): the prompt stays visible and only the response is masked.
prompt_len, resp_len, t = 3, 5, 0.4
m = rng.random((n, resp_len)) < t
print(f"\nSFT: prompt={prompt_len} tokens never masked, response={resp_len} tokens at t={t}:"
      f" expected response masks={resp_len * t:.2f}, empirical mean={m.sum(axis=1).mean():.4f}")
