"""Numbers for the "Exploration noise" example (Equation 7 and Section 7).

The paper gives theta = 0.15 and sigma = 0.2 and doesn't print the discretization.
This script uses the Euler step x <- x + theta (mu - x) dt + sigma sqrt(dt) eps with dt = 1.

Run: python3 scripts/ou_noise.py
"""

import numpy as np

theta, sigma, mu, dt = 0.15, 0.2, 0.0, 1.0
rng = np.random.default_rng(0)

n_paths, n_steps = 4000, 400
x = np.zeros(n_paths)
xs = np.empty((n_steps, n_paths))
for t in range(n_steps):
    x = x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_paths)
    xs[t] = x

# Stationary values for the dt = 1 Euler step: x' = (1 - theta) x + sigma eps.
rho = 1 - theta
var_stat = sigma**2 / (1 - rho**2)
print(f"lag-1 correlation, theory   = {rho:.3f}")
print(f"stationary std, theory      = {np.sqrt(var_stat):.4f}")

late = xs[100:]  # past the transient (0.85^100 is about 1e-7)
print(f"stationary std, empirical   = {late.std():.4f}")
corr = np.corrcoef(late[:-1].ravel(), late[1:].ravel())[0, 1]
print(f"lag-1 correlation, empirical = {corr:.3f}")

# Compare with independent noise of the same marginal std.
iid = rng.standard_normal((n_steps - 100, n_paths)) * np.sqrt(var_stat)
w = 20
ou_means = late[:w].mean(axis=0)
iid_means = iid[:w].mean(axis=0)
print(f"std of a 20-step average: OU = {ou_means.std():.4f}, independent = {iid_means.std():.4f}")

# A single 6-step path from rest, to compare by hand with the update rule.
x = 0.0
rng2 = np.random.default_rng(1)
eps = rng2.standard_normal(6)
path = []
for e in eps:
    x = x + theta * (mu - x) * dt + sigma * np.sqrt(dt) * e
    path.append(x)
print("six steps from 0 with seed 1:", np.round(path, 4))
print("noise terms sigma * eps     :", np.round(sigma * eps, 4))

# Noise-free decay: x0 = 1, sigma = 0 -> (1 - theta)^t.
print("noise-free decay from 1:", [round((1 - theta) ** t, 4) for t in range(1, 5)])
