"""Reverse-time SDE and probability-flow ODE on 1-D Gaussian data (Equations 6 and 13).

With x(0) ~ N(2, 0.5^2) the score of every marginal is known exactly, so no
network is needed. The script integrates Equation 6 with Euler-Maruyama and
Equation 13 with Euler from t = 1 to t = eps and prints sample statistics.
It also repeats the run from the same prior draw to show that the ODE is
deterministic and the SDE is not.

Run: python3 scripts/reverse_toy.py
"""

import numpy as np

BMIN, BMAX = 0.1, 20.0
M0, S0 = 2.0, 0.5
EPS = 1e-3  # Appendix D.4 uses eps = 1e-3 for VP SDEs


def beta(t):
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


def score(x, t):
    m = np.exp(-0.5 * int_beta(t))
    return -(x - m * M0) / (m**2 * S0**2 + 1 - m**2)


def run(n, N, mode, rng, x_T=None):
    x = rng.standard_normal(n) if x_T is None else x_T.copy()
    ts = np.linspace(1.0, EPS, N + 1)
    for k in range(N):
        t, dt = ts[k], ts[k + 1] - ts[k]  # dt < 0
        f = -0.5 * beta(t) * x
        if mode == "sde":
            drift = f - beta(t) * score(x, t)
            x = x + drift * dt + np.sqrt(beta(t) * -dt) * rng.standard_normal(n)
        else:
            x = x + (f - 0.5 * beta(t) * score(x, t)) * dt
    return x


m_eps = np.exp(-0.5 * int_beta(EPS))
v_eps = m_eps**2 * S0**2 + 1 - m_eps**2
print(f"Target p_eps: mean {m_eps * M0:.4f}, std {np.sqrt(v_eps):.4f}   (p_0: mean {M0}, std {S0})")
for N in (10, 50, 1000):
    for mode in ("sde", "ode"):
        x = run(100_000, N, mode, np.random.default_rng(1))
        print(f"N = {N:4d}  {mode}:  mean {x.mean():.4f}  std {x.std():.4f}")

rng = np.random.default_rng(2)
x_T = rng.standard_normal(5)
print()
print("Same five prior draws, two runs each (N = 1000):")
print("  ode run 1:", np.round(run(5, 1000, "ode", np.random.default_rng(3), x_T), 4))
print("  ode run 2:", np.round(run(5, 1000, "ode", np.random.default_rng(4), x_T), 4))
print("  sde run 1:", np.round(run(5, 1000, "sde", np.random.default_rng(3), x_T), 4))
print("  sde run 2:", np.round(run(5, 1000, "sde", np.random.default_rng(4), x_T), 4))
