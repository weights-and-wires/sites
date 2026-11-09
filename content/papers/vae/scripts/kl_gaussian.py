"""Closed-form Gaussian KL (Appendix B) against a Monte Carlo estimate.

Run: python3 scripts/kl_gaussian.py
"""

import numpy as np

mu = np.array([0.5, -1.0])
sigma = np.array([0.8, 1.5])

terms = 0.5 * (1 + np.log(sigma**2) - mu**2 - sigma**2)
print("per-dimension -KL terms =", np.round(terms, 4).tolist())
print(f"-KL (Appendix B)        = {terms.sum():.4f}")
print(f"KL                      = {-terms.sum():.4f}")

rng = np.random.default_rng(0)
n = 1_000_000
z = mu + sigma * rng.standard_normal((n, 2))
log_q = -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + ((z - mu) / sigma) ** 2, axis=1)
log_p = -0.5 * np.sum(np.log(2 * np.pi) + z**2, axis=1)
print(f"Monte Carlo KL ({n} samples) = {np.mean(log_q - log_p):.4f}")

print()
print("Same mu, sigma = 1 in both dimensions (variance term vanishes)")
print(f"KL = {0.5 * np.sum(mu**2):.4f}")
print("mu = 0, sigma = 1 (q equals the prior): KL =", 0.5 * (0 + 1 - 1 - 0))
