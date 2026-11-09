"""Conditional Gaussian paths and their vector fields (Section 4, Theorem 3).

Notation follows the paper: x0 is a noise sample, x1 is a data sample, t = 0 is
noise and t = 1 is data. This script prints every number in the page's examples
for the optimal-transport (OT) path (Equations 20 to 23) and the variance-
preserving (VP) diffusion path (Equations 18 and 19).

Run: python3 scripts/conditional_paths.py
"""

import numpy as np

SIGMA_MIN = 0.1
BMIN, BMAX = 0.1, 20.0  # beta schedule of Appendix E.1


# ---- OT path (Equations 20 to 23) -------------------------------------------
def ot_mu(t, x1):
    return t * x1


def ot_sigma(t):
    return 1.0 - (1.0 - SIGMA_MIN) * t


def ot_psi(t, x0, x1):  # Equation 22
    return ot_sigma(t) * x0 + t * x1


def ot_u(t, x, x1):  # Equation 21
    return (x1 - (1.0 - SIGMA_MIN) * x) / ot_sigma(t)


# ---- VP path (Equations 18 and 19) ------------------------------------------
def T(s):
    return BMIN * s + 0.5 * s**2 * (BMAX - BMIN)


def Tp(s):
    return BMIN + s * (BMAX - BMIN)


def alpha(s):
    return np.exp(-0.5 * T(s))


def vp_mu(t, x1):
    return alpha(1 - t) * x1


def vp_sigma(t):
    return np.sqrt(1 - alpha(1 - t) ** 2)


def vp_u(t, x, x1):  # Equation 19
    s = 1 - t
    return -Tp(s) / 2 * (np.exp(-T(s)) * x - np.exp(-0.5 * T(s)) * x1) / (1 - np.exp(-T(s)))


def vp_u_theorem3(t, x, x1, h=1e-6):  # Equation 15 with finite-difference derivatives
    mu = vp_mu(t, x1)
    dmu = (vp_mu(t + h, x1) - vp_mu(t - h, x1)) / (2 * h)
    sg = vp_sigma(t)
    dsg = (vp_sigma(t + h) - vp_sigma(t - h)) / (2 * h)
    return dsg / sg * (x - mu) + dmu


def main():
    x1, x0 = 2.0, -1.0
    print(f"OT path, sigma_min = {SIGMA_MIN}, x1 = {x1}, x0 = {x0}")
    print("t     mu_t    sigma_t  psi_t(x0)  u_t(psi|x1)  d/dt psi (finite diff)")
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        psi = ot_psi(t, x0, x1)
        h = 1e-6
        fd = (ot_psi(t + h, x0, x1) - ot_psi(t - h, x0, x1)) / (2 * h)
        print(f"{t:<5} {ot_mu(t, x1):<7.4f} {ot_sigma(t):<8.4f} {psi:<10.4f} {ot_u(t, psi, x1):<12.4f} {fd:.4f}")
    print(f"Equation 23 target x1 - (1 - sigma_min) x0 = {x1 - (1 - SIGMA_MIN) * x0:.4f}")

    print("\nCFM loss for one pair at t = 0.5 (Equation 23):")
    t = 0.5
    xt = ot_psi(t, x0, x1)
    target = x1 - (1 - SIGMA_MIN) * x0
    for name, v in (("v = 0", 0.0), ("v = 1", 1.0), ("v = target", target)):
        print(f"  {name:<11} loss = {(v - target) ** 2:.4f}")
    print(f"  x_t = {xt:.4f}")

    print(f"\nVP path, beta from {BMIN} to {BMAX}, x1 = {x1}")
    print("t     alpha_{1-t}  mu_t     sigma_t  u_t(mu_t + sigma_t x0 | x1)  Theorem 3 check")
    for t in (0.25, 0.5, 0.75):
        mu, sg = vp_mu(t, x1), vp_sigma(t)
        x = sg * x0 + mu
        print(f"{t:<5} {alpha(1 - t):<12.6f} {mu:<8.4f} {sg:<8.4f} {vp_u(t, x, x1):<29.4f} {vp_u_theorem3(t, x, x1):.4f}")

    print("\nStraightness of the conditional path (arc length / endpoint distance)")
    ts = np.linspace(0.0, 1.0 - 1e-4, 20001)
    for x0v in (-1.0, 0.0, 1.0):
        ot = ot_psi(ts, x0v, x1)
        vp = vp_sigma(ts) * x0v + vp_mu(ts, x1)
        d_ot = abs(ot[-1] - ot[0])
        d_vp = abs(vp[-1] - vp[0])
        len_ot = np.abs(np.diff(ot)).sum()
        len_vp = np.abs(np.diff(vp)).sum()
        print(f"  x0 = {x0v:+.1f}: OT {len_ot / d_ot:.4f}   VP {len_vp / d_vp:.4f}   (VP max x = {vp.max():.4f}, VP end = {vp[-1]:.4f})")


if __name__ == "__main__":
    main()
