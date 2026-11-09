#!/usr/bin/env python3
"""Latent sampling, rescaling, and noising (Section 3.1, Appendix B Eq. 4, Appendix G).

All numbers on the page about these steps come from this script.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# 1. Sample from a KL-regularized encoder output: z = mu + sigma * eps (Appendix G).
mu = np.array([1.0, -2.0])
logvar = np.array([0.0, np.log(4.0)])      # variances 1 and 4
eps = np.array([0.5, -1.0])
z = mu + np.exp(0.5 * logvar) * eps
print("z =", z)                            # [1.5, -4.0]

# 2. KL(N(mu, sigma^2) || N(0, 1)) per component, and its weighted total.
kl = 0.5 * (mu**2 + np.exp(logvar) - 1 - logvar)
print("KL per component =", kl, " sum =", kl.sum())
print("weighted by 1e-6 =", 1e-6 * kl.sum())

# 3. Rescaling: one scalar std over the b, c, h, w axes of the first batch.
rng = np.random.default_rng(0)
batch = 5.0 * rng.standard_normal((4, 3, 8, 8)) + 1.0
sigma_hat = np.sqrt(np.mean((batch - batch.mean()) ** 2))
print("sigma_hat =", round(float(sigma_hat), 4))
rescaled = batch / sigma_hat
print("rescaled std =", round(float(rescaled.std()), 4))

# 4. Forward noising in latent space, written with alpha_t and sigma_t (Eq. 4).
#    Variance preserving: alpha_t^2 + sigma_t^2 = 1.
alpha_t, sigma_t = 0.6, 0.8
print("SNR(t) = alpha^2 / sigma^2 =", round(alpha_t**2 / sigma_t**2, 4))
for std in (1.0, 5.0):
    z0 = std * rng.standard_normal((4, 3, 8, 8))
    signal = (alpha_t * z0).var()
    noise = sigma_t**2
    print(f"latent std {std}: signal variance / noise variance at t = {signal / noise:.3f}")
