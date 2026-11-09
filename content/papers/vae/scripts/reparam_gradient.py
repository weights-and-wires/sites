"""Compare the score-function estimator with the reparameterized estimator.

Target: d/d(mu) and d/d(sigma) of E_{z ~ N(mu, sigma^2)}[z^2] = mu^2 + sigma^2,
so the true gradients are 2 mu and 2 sigma.

Run: python3 scripts/reparam_gradient.py
"""

import numpy as np

mu, sigma = 1.0, 1.0
rng = np.random.default_rng(0)
n = 200_000
eps = rng.standard_normal(n)
z = mu + sigma * eps
f = z**2

# Naive estimator from Section 2.2: f(z) * grad_mu log q(z).
score_mu = f * (z - mu) / sigma**2
score_sigma = f * ((z - mu) ** 2 / sigma**3 - 1 / sigma)
# Reparameterized estimator (Section 2.4): differentiate f(mu + sigma * eps).
rep_mu = 2 * z
rep_sigma = 2 * z * eps

print(f"true d/dmu = {2 * mu}, true d/dsigma = {2 * sigma}")
print(f"score function: mean d/dmu = {score_mu.mean():.3f}, std of one sample = {score_mu.std():.3f}")
print(f"score function: mean d/dsigma = {score_sigma.mean():.3f}, std of one sample = {score_sigma.std():.3f}")
print(f"reparameterized: mean d/dmu = {rep_mu.mean():.3f}, std of one sample = {rep_mu.std():.3f}")
print(f"reparameterized: mean d/dsigma = {rep_sigma.mean():.3f}, std of one sample = {rep_sigma.std():.3f}")
print(f"variance ratio for mu = {score_mu.var() / rep_mu.var():.1f}")
