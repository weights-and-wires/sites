"""Run Algorithm 2 with an exact epsilon predictor for 1-D Gaussian data.

For x0 ~ N(m, s^2), x_t is Gaussian and E[eps | x_t] has a closed form, so no
network is needed. The script prints the sample mean and standard deviation for
both variance choices in Section 3.2 and for a sampler that never adds noise.

Run: python3 scripts/oracle_sampler.py
"""

import time

import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)
alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
beta_tilde = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas

m, s = 2.0, 0.5


def eps_oracle(x, t):
    ab = alpha_bars[t - 1]
    return np.sqrt(1 - ab) * (x - np.sqrt(ab) * m) / (ab * s**2 + 1 - ab)


def sample(n, sigma2, rng, noise=True):
    x = rng.standard_normal(n)
    for t in range(T, 0, -1):
        z = rng.standard_normal(n) if (t > 1 and noise) else 0.0
        x = (x - betas[t - 1] / np.sqrt(1 - alpha_bars[t - 1]) * eps_oracle(x, t)) / np.sqrt(alphas[t - 1]) + np.sqrt(sigma2[t - 1]) * z
    return x


start = time.time()
for name, sigma2, noise in (("sigma^2 = beta_t", betas, True), ("sigma^2 = beta_tilde_t", beta_tilde, True), ("no noise (z = 0)", betas, False)):
    x = sample(20_000, sigma2, np.random.default_rng(0), noise)
    print(f"{name:24s} mean {x.mean():.3f}  std {x.std():.3f}   (target mean {m}, std {s})")
print(f"elapsed {time.time() - start:.2f} s")
