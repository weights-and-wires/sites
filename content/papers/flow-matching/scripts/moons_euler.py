"""Euler sampling of the exact optimum of flow matching on a two-moons dataset.

With a finite dataset of N points, the CFM minimizer is the marginal field of
Equation 8 with q the empirical distribution, so no network is needed: this is
the vector field an infinitely flexible model would learn. The script
integrates it from t = 0 (noise) to t = 1 (data) with Euler steps and
compares 5, 10, 20, and 50 steps against a 1000-step midpoint reference that
starts from the same noise.

Run: python3 scripts/moons_euler.py
"""

import numpy as np

SIGMA_MIN = 0.05
N_DATA = 200
N_SAMPLES = 1000
SEED = 0


def make_moons(n, noise, rng):
    """Two interleaved half circles, like sklearn.datasets.make_moons."""
    k = n // 2
    a = np.linspace(0, np.pi, k)
    upper = np.stack([np.cos(a), np.sin(a)], axis=1)
    lower = np.stack([1 - np.cos(a), 0.5 - np.sin(a)], axis=1)
    pts = np.concatenate([upper, lower])
    return pts + noise * rng.normal(size=pts.shape)


def ot_sigma(t):
    return 1.0 - (1.0 - SIGMA_MIN) * t


def marginal_field(x, t, data):
    """Equation 8 for the OT path: posterior-weighted average of Equation 21."""
    s = ot_sigma(t)
    d2 = np.sum((x[:, None, :] - t * data[None]) ** 2, axis=-1)  # (n, N)
    logw = -0.5 * d2 / s**2
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    u = (data[None] - (1 - SIGMA_MIN) * x[:, None, :]) / s  # (n, N, 2)
    return np.sum(w[:, :, None] * u, axis=1)


def euler(field, x0, n_steps):
    x = x0.copy()
    for k in range(n_steps):
        t = k / n_steps
        x = x + field(x, t) / n_steps
    return x


def midpoint(field, x0, n_steps):
    x = x0.copy()
    h = 1.0 / n_steps
    for k in range(n_steps):
        t = k * h
        x = x + h * field(x + 0.5 * h * field(x, t), t + 0.5 * h)
    return x


def main():
    rng = np.random.default_rng(SEED)
    data = make_moons(N_DATA, 0.05, rng)
    x0 = rng.normal(size=(N_SAMPLES, 2))
    field = lambda x, t: marginal_field(x, t, data)

    ref = midpoint(field, x0, 1000)

    def dist_to_data(x):
        d = np.sqrt(np.sum((x[:, None, :] - data[None]) ** 2, axis=-1))
        return d.min(axis=1)

    print(f"data: {N_DATA} points, sigma_min = {SIGMA_MIN}, {N_SAMPLES} noise samples (seed {SEED})")
    print(f"reference (1000 midpoint steps): mean distance to nearest data point = {dist_to_data(ref).mean():.4f}")
    print()
    print("Euler steps | MSE to reference | mean distance to nearest data point")
    for n in (5, 10, 20, 50):
        x1 = euler(field, x0, n)
        mse = np.mean(np.sum((x1 - ref) ** 2, axis=1))
        print(f"{n:>11} | {mse:16.5f} | {dist_to_data(x1).mean():.4f}")

    print()
    print("Midpoint steps (2 field calls each) | MSE to reference")
    for n in (3, 5, 10, 25):
        x1 = midpoint(field, x0, n)
        mse = np.mean(np.sum((x1 - ref) ** 2, axis=1))
        print(f"{n:>11} steps, {2 * n:>2} calls | {mse:.5f}")

    print()
    print("Moment check for the 50-step Euler samples vs the data")
    x1 = euler(field, x0, 50)
    print(f"  data mean {np.round(data.mean(0), 3)}, samples mean {np.round(x1.mean(0), 3)}")
    print(f"  data std  {np.round(data.std(0), 3)}, samples std  {np.round(x1.std(0), 3)}")


if __name__ == "__main__":
    main()
