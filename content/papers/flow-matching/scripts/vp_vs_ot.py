"""Compare the OT path and the VP diffusion path on the two-moons optimum.

Both paths use the exact marginal field of Equation 8 for the same 200 data
points, so the only difference is the conditional path: OT (Equations 20 and 21)
or VP (Equations 18 and 19). Each is integrated with Euler steps and scored by
the mean distance from a sample to the nearest data point, a rough quality
measure where lower is better. VP stops at t = 1 - 1e-3, because sigma_t = 0 at
t = 1 for the VP path with alpha_0 = 1.

Run: python3 scripts/vp_vs_ot.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moons_euler import SIGMA_MIN, make_moons  # noqa: E402

BMIN, BMAX = 0.1, 20.0
EPS = 1e-3


def T(s):
    return BMIN * s + 0.5 * s**2 * (BMAX - BMIN)


def Tp(s):
    return BMIN + s * (BMAX - BMIN)


def ot_field(x, t, data):
    s = 1.0 - (1.0 - SIGMA_MIN) * t
    d2 = np.sum((x[:, None, :] - t * data[None]) ** 2, axis=-1)
    w = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / s**2)
    w /= w.sum(axis=1, keepdims=True)
    u = (data[None] - (1 - SIGMA_MIN) * x[:, None, :]) / s
    return np.sum(w[:, :, None] * u, axis=1)


def vp_field(x, t, data):
    s_ = 1.0 - t
    a = np.exp(-0.5 * T(s_))
    var = 1.0 - a**2
    d2 = np.sum((x[:, None, :] - a * data[None]) ** 2, axis=-1)
    w = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / var)
    w /= w.sum(axis=1, keepdims=True)
    u = -Tp(s_) / 2 * (a**2 * x[:, None, :] - a * data[None]) / var  # Equation 19
    return np.sum(w[:, :, None] * u, axis=1)


def euler(field, x0, n_steps, t_end):
    x = x0.copy()
    h = t_end / n_steps
    for k in range(n_steps):
        x = x + h * field(x, k * h)
    return x


def main():
    rng = np.random.default_rng(0)
    data = make_moons(200, 0.05, rng)
    x0 = rng.normal(size=(500, 2))

    def dist(x):
        return np.sqrt(np.sum((x[:, None, :] - data[None]) ** 2, axis=-1)).min(axis=1).mean()

    print("mean distance to nearest data point (lower is better), 500 samples, Euler")
    print("steps |     OT |     VP")
    for n in (5, 10, 20, 50, 200):
        ot = dist(euler(lambda x, t: ot_field(x, t, data), x0, n, 1.0))
        vp = dist(euler(lambda x, t: vp_field(x, t, data), x0, n, 1.0 - EPS))
        print(f"{n:>5} | {ot:.4f} | {vp:.4f}")


if __name__ == "__main__":
    main()
