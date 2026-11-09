"""Sample a 2-D Gaussian mixture with the probability-flow ODE (Equation 13).

The data are a three-component mixture with isotropic components of standard
deviation 0.3. Under the VP-SDE every p_t is a mixture too, so the exact score
is available and no network is needed. The script integrates Equation 13 from
t = 1 to t = eps = 1e-3 with Euler steps, and compares the result with the
reverse-time SDE (Equation 6) and with the true mixture.

Run: python3 scripts/toy2d_pf_ode.py
"""

import time

import numpy as np

BMIN, BMAX = 0.1, 20.0
EPS = 1e-3
W = np.array([0.5, 0.3, 0.2])
MU = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
S = 0.3


def beta(t):
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


def score(x, t):
    """Exact score of p_t for x of shape (n, 2)."""
    m = np.exp(-0.5 * int_beta(t))
    v = m**2 * S**2 + 1 - m**2
    d = x[:, None, :] - m * MU[None]  # (n, K, 2)
    logw = np.log(W)[None] - 0.5 * np.sum(d**2, axis=-1) / v
    r = np.exp(logw - logw.max(axis=1, keepdims=True))
    r /= r.sum(axis=1, keepdims=True)
    return -np.sum(r[:, :, None] * d, axis=1) / v


def integrate(x, N, mode, rng):
    ts = np.linspace(1.0, EPS, N + 1)
    for k in range(N):
        t, dt = ts[k], ts[k + 1] - ts[k]
        f = -0.5 * beta(t) * x
        if mode == "ode":
            x = x + (f - 0.5 * beta(t) * score(x, t)) * dt
        else:
            x = x + (f - beta(t) * score(x, t)) * dt + np.sqrt(beta(t) * -dt) * rng.standard_normal(x.shape)
    return x


def report(name, x):
    k = np.argmin(np.sum((x[:, None, :] - MU[None]) ** 2, axis=-1), axis=1)
    frac = np.bincount(k, minlength=3) / len(x)
    print(f"{name}")
    print(f"  mode weights {np.round(frac, 3)}   (target {W})")
    for j in range(3):
        c = x[k == j]
        print(f"  mode {j}: mean {np.round(c.mean(axis=0), 3)}  std {np.round(c.std(axis=0), 3)}  (target mean {MU[j]}, std {S})")


rng = np.random.default_rng(0)
n = 5000
x_T = rng.standard_normal((n, 2))
start = time.time()
for N in (20, 500):
    report(f"ODE, N = {N}", integrate(x_T.copy(), N, "ode", rng))
report("SDE, N = 500", integrate(x_T.copy(), 500, "sde", rng))
print(f"elapsed {time.time() - start:.2f} s")

# Determinism and a coarse invertibility check: ODE from the same x_T gives the same output.
a = integrate(x_T[:5].copy(), 500, "ode", np.random.default_rng(10))
b = integrate(x_T[:5].copy(), 500, "ode", np.random.default_rng(11))
print("ODE outputs identical for identical x_T:", bool(np.array_equal(a, b)))
