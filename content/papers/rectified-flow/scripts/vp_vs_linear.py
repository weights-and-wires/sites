#!/usr/bin/env python3
"""The interpolation coefficients of the VP ODE against the linear rectified flow.

Section 2.3.1 writes every path as X_t = alpha_t X1 + beta_t X0 (or beta_t xi).
The rectified flow uses alpha_t = t and beta_t = 1 - t. The VP ODE uses
(Equation 7, with a = 19.9 and b = 0.1)

    alpha_t = exp(-a (1 - t)^2 / 4 - b (1 - t) / 2),    beta_t = sqrt(1 - alpha_t^2)  (Equation 8).

The script prints both at a few times, the fraction of the way that alpha_t has
travelled, and the largest distance of the coefficient curve from the straight
segment alpha + beta = 1. The straight segment is what beta_t = 1 - alpha_t gives.

Run from the repository root:

    python3 content/papers/rectified-flow/scripts/vp_vs_linear.py
"""

import numpy as np

A_COEF, B_COEF = 19.9, 0.1


def vp_alpha(t):
    return np.exp(-0.25 * A_COEF * (1.0 - t) ** 2 - 0.5 * B_COEF * (1.0 - t))


def main():
    print("      linear           VP ODE (Equations 7 and 8)")
    print("  t   alpha  beta     alpha    beta    alpha + beta")
    for t in (0.25, 0.5, 0.75, 0.9):
        a = float(vp_alpha(t))
        b = float(np.sqrt(1.0 - a * a))
        print(f"{t:4.2f}   {t:.2f}  {1 - t:.2f}     {a:.4f}   {b:.4f}   {a + b:.4f}")

    t = np.linspace(0.0, 1.0, 100001)
    a = vp_alpha(t)
    b = np.sqrt(1.0 - a * a)
    gap = (a + b - 1.0) / np.sqrt(2.0)
    print(f"\nlargest distance of the VP curve (alpha_t, beta_t) from the segment alpha + beta = 1: {gap.max():.4f}")
    print(f"alpha_0 = {a[0]:.2e} (the paper's condition alpha_0 = 0 holds only approximately)")
    half = float(vp_alpha(0.5))
    print(f"at t = 0.5 the VP ODE has covered {half:.1%} of the way in alpha; the linear path has covered 50.0%")
    first_half = (a[50000] - a[0]) / (a[-1] - a[0])
    print(f"share of the total change in alpha that happens in the first half of time: VP {first_half:.1%}, linear 50.0%")


if __name__ == "__main__":
    main()
