"""Euler-Maruyama simulation of the forward VP-SDE (Equation 11) against Equation 33.

Start from x(0) = 2, simulate with N steps of size 1/N, and compare the sample
mean and variance at several times with the closed form.

Run: python3 scripts/em_forward.py
"""

import numpy as np

BMIN, BMAX = 0.1, 20.0


def beta(t):
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


x0 = 2.0
n_samples = 400_000
rng = np.random.default_rng(0)

for N in (10, 100, 1000):
    dt = 1.0 / N
    x = np.full(n_samples, x0)
    marks = {N // 10: 0.1, N // 2: 0.5, N: 1.0}
    out = []
    for k in range(N):
        t = k * dt
        x = x - 0.5 * beta(t) * x * dt + np.sqrt(beta(t) * dt) * rng.standard_normal(n_samples)
        if (k + 1) in marks:
            tt = marks[k + 1]
            want_m = x0 * np.exp(-0.5 * int_beta(tt))
            want_v = 1 - np.exp(-int_beta(tt))
            out.append((tt, x.mean(), want_m, x.var(), want_v))
    print(f"N = {N}")
    for tt, m, wm, v, wv in out:
        print(f"  t = {tt:3.1f}  mean {m:7.4f} (theory {wm:7.4f})   var {v:7.4f} (theory {wv:7.4f})")
