"""Check that the SDE and the probability-flow ODE share one Fokker-Planck equation.

For x(0) ~ N(2, 0.5^2) and the VP-SDE, p_t is Gaussian in closed form. The
script estimates dp/dt by a central difference in t and compares it with
  (a) the Fokker-Planck right side of the forward SDE (Appendix D.1, Equation 35 with G = g)
  (b) the continuity equation of the probability-flow ODE, -d/dx [(f - g^2 score / 2) p]
using finite differences in x. Both residuals should be near zero.

Run: python3 scripts/fokker_planck.py
"""

import numpy as np

BMIN, BMAX = 0.1, 20.0
M0, S0 = 2.0, 0.5


def beta(t):
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


def p(x, t):
    m = np.exp(-0.5 * int_beta(t))
    v = m**2 * S0**2 + 1 - m**2
    return np.exp(-0.5 * (x - m * M0) ** 2 / v) / np.sqrt(2 * np.pi * v)


def score(x, t):
    m = np.exp(-0.5 * int_beta(t))
    return -(x - m * M0) / (m**2 * S0**2 + 1 - m**2)


x = np.linspace(-4, 5, 4001)
h = x[1] - x[0]
dx_d = lambda y: np.gradient(y, h)

for t in (0.05, 0.3, 0.7):
    dt = 1e-5
    dpdt = (p(x, t + dt) - p(x, t - dt)) / (2 * dt)
    pt = p(x, t)
    f = -0.5 * beta(t) * x
    fp_rhs = -dx_d(f * pt) + 0.5 * beta(t) * dx_d(dx_d(pt))
    flow_rhs = -dx_d((f - 0.5 * beta(t) * score(x, t)) * pt)
    sl = slice(100, -100)
    scale = np.max(np.abs(dpdt[sl]))
    print(
        f"t = {t:4.2f}  max|dp/dt| {scale:.4f}   "
        f"SDE residual {np.max(np.abs(dpdt - fp_rhs)[sl]):.2e}   "
        f"ODE residual {np.max(np.abs(dpdt - flow_rhs)[sl]):.2e}"
    )
