"""One shared linear encoder against the per-datapoint optimum.

Model: p(z) = N(0, 1), p(x|z) = N(x; z, 1). The true posterior is N(x/2, 1/2), so an
encoder mu(x) = w x, sigma^2 = 0.5 can be exact for every x at once with w = 0.5.

Run: python3 scripts/amortized_example.py
"""

import numpy as np


def kl_gauss(m1, v1, m2, v2):
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)


xs = np.array([-2.0, 1.0, 3.0])
print("gap = log p(x) - ELBO = KL(q || p(z|x))")
for w in (0.5, 0.3):
    gaps = kl_gauss(w * xs, 0.5, xs / 2, 0.5)
    print(f"w = {w}: gaps for x = {xs.tolist()} -> {np.round(gaps, 4).tolist()}")

# Fit w and log sigma^2 by SGD on the negative ELBO with reparameterized gradients.
rng = np.random.default_rng(0)
w, c = 0.0, 0.0  # c = log sigma^2
lr, batch, steps = 0.02, 128, 3000
for _ in range(steps):
    x = np.sqrt(2.0) * rng.standard_normal(batch)  # x ~ p(x) = N(0, 2)
    eps = rng.standard_normal(batch)
    sigma = np.exp(0.5 * c)
    mu = w * x
    z = mu + sigma * eps
    d_mu = mu - (x - z)
    d_c = 0.5 * (np.exp(c) - 1) - (x - z) * eps * 0.5 * sigma
    w -= lr * np.mean(d_mu * x)
    c -= lr * np.mean(d_c)
print(f"\nafter {steps} steps: w = {w:.3f} (optimum 0.5), log sigma^2 = {c:.3f} (optimum {np.log(0.5):.3f})")
