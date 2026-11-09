"""Compute the numbers for the "normalize across features" example.

Uses the paper's Equation 3 (no epsilon) and Equation 4's form h = f(g/sigma * (a - mu) + b)
with f = identity, so the output is the normalized, scaled, and shifted vector.

Run: python3 scripts/layernorm_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

a = np.array([1.0, 2.0, 3.0, 6.0])  # summed inputs a^l for one example, H = 4
H = a.size
mu = a.sum() / H
sigma = np.sqrt(((a - mu) ** 2).sum() / H)
a_hat = (a - mu) / sigma
print(f"a      = {a.tolist()}")
print(f"mu     = {mu:.4f}")
print(f"var    = {sigma**2:.4f}")
print(f"sigma  = {sigma:.4f}")
print(f"a_hat  = {a_hat}")
print(f"mean(a_hat) = {a_hat.mean():.4f}, var(a_hat) = {a_hat.var():.4f}")

g = np.array([1.0, 2.0, 1.0, 0.5])
b = np.array([0.0, 0.5, 0.0, -1.0])
print(f"g      = {g.tolist()}, b = {b.tolist()}")
print(f"g * a_hat + b = {g * a_hat + b}")

# Equation 7: rescaling and shifting one example leaves the output unchanged.
a2 = 10 * a + 5
mu2 = a2.mean()
sigma2 = np.sqrt(((a2 - mu2) ** 2).mean())
print()
print(f"a' = 10 a + 5 = {a2.tolist()}")
print(f"mu' = {mu2:.4f}, sigma' = {sigma2:.4f}")
print(f"a_hat' = {(a2 - mu2) / sigma2}")
print(f"same output: {np.allclose((a2 - mu2) / sigma2, a_hat)}")
