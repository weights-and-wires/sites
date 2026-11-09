"""Check Equations 1 to 3 and the SGVB estimator on a model with a known posterior.

Model: p(z) = N(0, 1), p(x|z) = N(x; z, 1). Then p(x) = N(x; 0, 2) and the true
posterior is p(z|x) = N(z; x/2, 1/2), so every term of Equation 1 has a closed form.

Run: python3 scripts/elbo_identity.py
"""

import numpy as np

x = 1.0


def log_normal(v, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - (v - mean) ** 2 / (2 * var)


def kl_gauss(m1, v1, m2, v2):
    """KL(N(m1, v1) || N(m2, v2)) for scalars."""
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)


def elbo_exact(x, mu, var):
    """Equation 3 with both terms in closed form."""
    kl_prior = kl_gauss(mu, var, 0.0, 1.0)
    exp_loglik = -0.5 * np.log(2 * np.pi) - 0.5 * ((x - mu) ** 2 + var)
    return -kl_prior + exp_loglik


log_px = log_normal(x, 0.0, 2.0)
post_mu, post_var = x / 2, 0.5
print(f"log p(x)             = {log_px:.4f}")

for name, mu, var in [("q = N(0.2, 0.8^2)", 0.2, 0.64), ("q = true posterior", post_mu, post_var)]:
    kl_post = kl_gauss(mu, var, post_mu, post_var)
    elbo = elbo_exact(x, mu, var)
    print(f"\n{name}")
    print(f"  KL(q || p(z|x))    = {kl_post:.4f}")
    print(f"  ELBO (Equation 3)  = {elbo:.4f}")
    print(f"  KL + ELBO          = {kl_post + elbo:.4f}   (Equation 1: equals log p(x))")

# SGVB estimators with L = 100000 samples (Equations 6 and 7).
rng = np.random.default_rng(0)
mu, var = 0.2, 0.64
eps = rng.standard_normal(100_000)
z = mu + np.sqrt(var) * eps
est_a = np.mean(log_normal(x, z, 1.0) + log_normal(z, 0.0, 1.0) - log_normal(z, mu, var))
est_b = -kl_gauss(mu, var, 0.0, 1.0) + np.mean(log_normal(x, z, 1.0))
print("\nSGVB estimators for q = N(0.2, 0.8^2), L = 100000")
print(f"  estimator A (Eq 6) = {est_a:.4f}")
print(f"  estimator B (Eq 7) = {est_b:.4f}")
print(f"  exact ELBO         = {elbo_exact(x, mu, var):.4f}")

# Single-sample (L = 1) spread of each estimator.
e1 = rng.standard_normal(100_000)
z1 = mu + np.sqrt(var) * e1
a1 = log_normal(x, z1, 1.0) + log_normal(z1, 0.0, 1.0) - log_normal(z1, mu, var)
b1 = -kl_gauss(mu, var, 0.0, 1.0) + log_normal(x, z1, 1.0)
print("\nStandard deviation over 100000 single-sample (L = 1) estimates")
print(f"  estimator A        = {a1.std():.4f}")
print(f"  estimator B        = {b1.std():.4f}")

# At the true posterior, estimator A has no variance at all: log p(x, z) - log q(z|x)
# equals log p(x) for every z.
pm, pv = post_mu, post_var
zp = pm + np.sqrt(pv) * e1
ap = log_normal(x, zp, 1.0) + log_normal(zp, 0.0, 1.0) - log_normal(zp, pm, pv)
bp = -kl_gauss(pm, pv, 0.0, 1.0) + log_normal(x, zp, 1.0)
print("\nSame comparison with q equal to the true posterior")
print(f"  estimator A        = {ap.std():.4f}")
print(f"  estimator B        = {bp.std():.4f}")
