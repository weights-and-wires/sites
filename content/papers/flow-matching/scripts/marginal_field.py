"""The marginal vector field and the CFM equivalence (Theorems 1 and 2).

One-dimensional data with two equally likely points, x1 = -2 and x1 = +2, and the
OT conditional path with sigma_min = 0.1 (Equations 20 and 21). The script:

1. builds the marginal field u_t(x) with Equation 8 and prints it at t = 0.5;
2. checks that u_t generates the marginal path p_t of Equation 6 by testing the
   continuity equation (Appendix B, Equation 26) with finite differences;
3. shows that L_CFM - L_FM (Equations 9 and 5) is the same constant for every
   candidate field v, which is Theorem 2.

Run: python3 scripts/marginal_field.py
"""

import numpy as np

SIGMA_MIN = 0.1
DATA = np.array([-2.0, 2.0])
Q = np.array([0.5, 0.5])


def sigma(t):
    return 1.0 - (1.0 - SIGMA_MIN) * t


def cond_density(t, x, x1):  # p_t(x | x1), Equation 10 with mu_t = t x1
    s = sigma(t)
    return np.exp(-0.5 * ((x - t * x1) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def cond_u(t, x, x1):  # Equation 21
    return (x1 - (1.0 - SIGMA_MIN) * x) / sigma(t)


def marginal_density(t, x):  # Equation 6
    return sum(q * cond_density(t, x, x1) for q, x1 in zip(Q, DATA))


def marginal_u(t, x):  # Equation 8
    num = sum(q * cond_u(t, x, x1) * cond_density(t, x, x1) for q, x1 in zip(Q, DATA))
    return num / marginal_density(t, x)


def main():
    t, x = 0.5, 0.5
    print(f"t = {t}, x = {x}, sigma_t = {sigma(t):.4f}")
    for x1 in DATA:
        print(f"  x1 = {x1:+.1f}: p_t(x|x1) = {cond_density(t, x, x1):.6f}, u_t(x|x1) = {cond_u(t, x, x1):.4f}")
    p = marginal_density(t, x)
    w = np.array([q * cond_density(t, x, x1) / p for q, x1 in zip(Q, DATA)])
    print(f"  p_t(x) = {p:.6f}; posterior weights q(x1|x) = {w[0]:.4f}, {w[1]:.4f}")
    print(f"  u_t(x) = {marginal_u(t, x):.4f}   (weights times conditional fields)")
    print(f"  at x = 0 by symmetry: u_t(0) = {marginal_u(t, 0.0):.4f}")

    # Continuity equation: dp/dt + d(p u)/dx should be 0 on a grid.
    xs = np.linspace(-4, 4, 41)
    h = 1e-5
    worst = 0.0
    for t0 in (0.2, 0.5, 0.8):
        dpdt = (marginal_density(t0 + h, xs) - marginal_density(t0 - h, xs)) / (2 * h)
        flux = lambda z: marginal_density(t0, z) * marginal_u(t0, z)
        dflux = (flux(xs + h) - flux(xs - h)) / (2 * h)
        worst = max(worst, np.max(np.abs(dpdt + dflux)))
    print(f"\nContinuity equation residual, max over t in (0.2, 0.5, 0.8) and 41 grid points: {worst:.2e}")

    # Theorem 2: L_CFM - L_FM does not depend on v.
    grid = np.linspace(-8, 8, 160001)
    dx = grid[1] - grid[0]
    t = 0.5
    candidates = {
        "v = 0": lambda z: np.zeros_like(z),
        "v = 1": lambda z: np.ones_like(z),
        "v = 2x": lambda z: 2 * z,
        "v = u_t": lambda z: marginal_u(t, z),
    }
    print(f"\nTheorem 2 at t = {t}: L_FM, L_CFM, and their difference")
    for name, v in candidates.items():
        l_fm = np.sum(marginal_density(t, grid) * (v(grid) - marginal_u(t, grid)) ** 2) * dx
        l_cfm = sum(
            q * np.sum(cond_density(t, grid, x1) * (v(grid) - cond_u(t, grid, x1)) ** 2) * dx
            for q, x1 in zip(Q, DATA)
        )
        print(f"  {name:<7} L_FM = {l_fm:9.4f}  L_CFM = {l_cfm:9.4f}  difference = {l_cfm - l_fm:.4f}")


if __name__ == "__main__":
    main()
