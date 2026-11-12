"""Numbers for the "Squashed Gaussian" example (Section 4.2 Equation 11, Appendix C Equations 20 and 21).

Run: python3 scripts/squash_logprob.py
"""

import math

import numpy as np


def log_normal(u, mu, std):
    return -0.5 * ((u - mu) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)


def squashed_log_prob(u, mu, std):
    corr = np.log(1.0 - np.tanh(u) ** 2)
    return np.sum(log_normal(u, mu, std) - corr, axis=-1)


def cdf(a, mu, std):
    """P(tanh(u) <= a) for u ~ N(mu, std^2)."""
    return 0.5 * (1 + math.erf((math.atanh(a) - mu) / (std * math.sqrt(2))))


# 1. One dimension, u = 0.5, mu = 0, std = 1.
u, mu, std = 0.5, 0.0, 1.0
ln = float(log_normal(u, mu, std))
corr = float(np.log(1 - np.tanh(u) ** 2))
print(f"u = {u}: a = tanh(u) = {np.tanh(u):.4f}")
print(f"  log N(u) = {ln:.4f}   log(1 - tanh^2 u) = {corr:.4f}   log pi(a) = {ln - corr:.4f}")

# 2. Check against the density of a = tanh(u) from its CDF, by a central difference.
a = math.tanh(u)
h = 1e-6
density = (cdf(a + h, mu, std) - cdf(a - h, mu, std)) / (2 * h)
print(f"  pi(a) from the CDF slope = {density:.6f}; exp(log pi) = {math.exp(ln - corr):.6f}")
print(f"  without the correction, exp(log N(u)) = {math.exp(ln):.6f}")

# 3. The density integrates to 1 over a in (-1, 1). On an a-grid, u = atanh(a) and 1 - tanh^2 u = 1 - a^2.
grid_a = np.linspace(-0.999999, 0.999999, 2000001)
da = grid_a[1] - grid_a[0]
log_pi = log_normal(np.arctanh(grid_a), mu, std) - np.log(1 - grid_a**2)
print(f"integral of pi(a) da = {np.sum(np.exp(log_pi)) * da:.6f}")
dens_naive = np.exp(log_normal(np.arctanh(grid_a), mu, std))
print(f"integral if you skip the correction = {np.sum(dens_naive) * da:.4f}")

# 4. Two dimensions: the log-probs add.
u2 = np.array([0.5, -1.0])
mu2 = np.array([0.0, 0.5])
std2 = np.array([1.0, 0.5])
print(f"2-D: log pi = {squashed_log_prob(u2, mu2, std2):.4f}")

# 5. Stability: for large |u|, 1 - tanh(u)^2 rounds to 0 in float64.
big = 25.0
print(f"u = 25: 1 - tanh^2 = {1 - np.tanh(big) ** 2}, naive correction = {np.log(1 - np.tanh(big) ** 2) if 1 - np.tanh(big) ** 2 > 0 else float('-inf')}")
stable = 2 * (np.log(2) - big - np.logaddexp(0, -2 * big))
print(f"  stable log(1 - tanh^2 u) = 2 (log 2 - u - softplus(-2u)) = {stable:.4f}")
print(f"  at u = 0.5 the stable form gives {2 * (np.log(2) - 0.5 - np.logaddexp(0, -1.0)):.4f}")
