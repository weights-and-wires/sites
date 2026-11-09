#!/usr/bin/env python3
"""Straightness, the identity behind Theorem 3.7, and few-step Euler on the 1-D example.

Same setup as rewire_1d.py: X0 ~ N(0, 1), X1 = -2 or +2 with probability 1/2,
independent. This script computes, by quadrature:

* the straightness S(Z^1) of the first rectified flow (Section 2.2, Equation 3),
* the non-crossing measure V((X0, X1)) (Theorem 3.6, Equation 12),
* both sides of Equation 13, E|X1 - X0|^2 - E|Z1 - Z0|^2 = S(Z) + V((X0, X1)),
* the second rectified flow, which is exactly straight, and
* the squared Wasserstein-2 error of N-step Euler against pi_1 for both flows.

The paper's Equation 13 is printed without the squares on the norms; the proof
takes c(x) = |x|^2, so this page uses the squares.

Run from the repository root:

    python3 content/papers/rectified-flow/scripts/straightness_1d.py
"""

import math

import numpy as np

from rewire_1d import A, gaussian_grid, graded_times, rk4_path, v_x


def v_second(x, t):
    """Rectified flow of the coupling (Z0, Z1) = (Z0, 2 sign(Z0)).

    There X1 - X0 = 2 s - x0 with x = (1 - t) x0 + 2 t s, s = sign(x0), so
    v(x, t) = (2 s - x) / (1 - t) on the support of X_t, which is |x| > 2 t.
    """
    return (A * np.sign(x) - x) / (1.0 - t)


def trapz(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(((y[1:] + y[:-1]) * np.diff(x)).sum() / 2.0)


def euler_end(v, z0, n_steps):
    z, h = z0.copy(), 1.0 / n_steps
    for k in range(n_steps):
        z = z + h * v(z, k * h)
    return z


def w2_to_data(z, z0):
    """Pair the sorted samples with the pi_1 quantiles at the same levels (z0 is ascending)."""
    return np.sort(z), A * np.sign(z0)


def main():
    z0, w = gaussian_grid()
    times = graded_times(2000)
    path, tt = rk4_path(v_x, z0, times)
    z1 = A * np.sign(z0)

    # S(Z^1) = integral over t of E|(Z1 - Z0) - v(Z_t, t)|^2  (Equation 3)
    integrand = np.array([(w * ((z1 - z0) - v_x(path[k], tt[k])) ** 2).sum() for k in range(len(tt))])
    s1 = float(trapz(integrand, tt))

    # V((X0, X1)) = integral over t of E Var(X1 - X0 | X_t)  (Equation 12)
    # Condition on X1 = +2 by symmetry: X_t = 2 t + (1 - t) X0, and Var(a | x) = 4 sech^2(2 t x / (1 - t)^2).
    def v_integrand(t):
        x = A * t + (1.0 - t) * z0
        e = np.exp(-2.0 * np.abs(A * t * x / (1.0 - t) ** 2))  # sech^2(u) = 4 e^(-2|u|) / (1 + e^(-2|u|))^2
        sech2 = 4.0 * e / (1.0 + e) ** 2
        return float((w * A**2 * sech2).sum()) / (1.0 - t) ** 2

    tv = graded_times(4000)[:-1]
    v_meas = float(trapz([v_integrand(t) for t in tv], tv))

    lhs = (A**2 + 1.0) - float((w * (z1 - z0) ** 2).sum())
    print("Equation 13: E|X1 - X0|^2 - E|Z1 - Z0|^2 = S(Z^1) + V((X0, X1))")
    print(f"  left side            : {lhs:.4f}   (4 sqrt(2/pi) = {4 * math.sqrt(2 / math.pi):.4f})")
    print(f"  S(Z^1)               : {s1:.4f}")
    print(f"  V((X0, X1))          : {v_meas:.4f}")
    print(f"  S(Z^1) + V           : {s1 + v_meas:.4f}")

    print("\nSecond rectification: the coupling (Z0, 2 sign(Z0)) has no crossings, so Rectify returns it.")
    x_t = lambda t: (1.0 - t) * z0 + t * z1
    path2, tt2 = rk4_path(v_second, z0, times)
    dev = max(float(np.abs(path2[k] - x_t(tt2[k])).max()) for k in range(len(tt2)))
    print(f"  largest gap between the flow and the straight line X_t: {dev:.2e}")
    integrand2 = np.array([(w * ((z1 - z0) - v_second(path2[k], tt2[k])) ** 2).sum() for k in range(len(tt2))])
    print(f"  S(Z^2) = {float(trapz(integrand2, tt2)):.2e}")

    print("\nN-step Euler error against pi_1, W2^2 (0 is exact; 4 is the collapse to the mean):")
    print("   N   1-rectified   2-rectified")
    for n in (1, 2, 4, 8, 16, 100):
        row = []
        for v in (v_x, v_second):
            zs, tgt = w2_to_data(euler_end(v, z0, n), z0)
            row.append(float((w * (zs - tgt) ** 2).sum()))
        print(f"{n:4d}   {row[0]:11.6f}   {row[1]:11.6f}")


if __name__ == "__main__":
    main()
