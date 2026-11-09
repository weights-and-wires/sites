"""Solver error on a Gaussian target, where the exact flow is known (Section 6.2 idea).

For data N(m, s^2) and the OT path, the marginal field is affine in x and the
exact flow is phi_1(x0) = m + sqrt(sigma_min^2 + s^2) x0. This script prints the
RMS error of Euler and midpoint steps against it, with m = 2, s = 0.5, and
sigma_min = 0. It also prints the numbers used in the page's self-check items.

Run: python3 scripts/gaussian_solvers.py
"""

import numpy as np

M, S, SM = 2.0, 0.5, 0.0


def field(x, t):
    sig = 1.0 - (1.0 - SM) * t
    big = np.sqrt(sig**2 + (t * S) ** 2)
    dbig = (-(1.0 - SM) * sig + t * S**2) / big
    return M + dbig / big * (x - t * M)


def euler(v, x0, n):
    x, h = x0.copy(), 1.0 / n
    for k in range(n):
        x = x + h * v(x, k * h)
    return x


def midpoint(v, x0, n):
    x, h = x0.copy(), 1.0 / n
    for k in range(n):
        t = k * h
        x = x + h * v(x + 0.5 * h * v(x, t), t + 0.5 * h)
    return x


def main():
    x0 = np.random.default_rng(0).normal(size=(200, 1))
    exact = M + np.sqrt(SM**2 + S**2) * x0
    rms = lambda out: float(np.sqrt(np.mean((out - exact) ** 2)))
    print("steps | Euler RMS | ratio | midpoint RMS | ratio | midpoint calls")
    prev_e = prev_m = None
    for n in (5, 10, 20, 40, 80):
        e, m = rms(euler(field, x0, n)), rms(midpoint(field, x0, n))
        re = "" if prev_e is None else f"{prev_e / e:.2f}"
        rm = "" if prev_m is None else f"{prev_m / m:.2f}"
        print(f"{n:>5} | {e:.3e} | {re:>5} | {m:.3e}    | {rm:>5} | {2 * n}")
        prev_e, prev_m = e, m
    print(f"\nEuler, 5 steps: sample std {euler(field, x0, 5).std():.4f}; 50 steps: {euler(field, x0, 50).std():.4f}; exact: {exact.std():.4f}")

    print("\nSelf-check numbers")
    x = 1.0
    for _ in range(2):
        x = x + 0.5 * x
    print(f"  Euler on dx/dt = x, x(0) = 1, two steps: {x:.4f}; exact e = {np.e:.4f}")
    print(f"  flow e^t x at t = ln 2: std of p_t = {np.exp(np.log(2)):.4f}")
    print(f"  Lemma 1: field at s = 0.3, x = 1 is -2x = {-2 * 1.0:+.1f}, so the reversed field is {+2 * 1.0:+.1f}")


if __name__ == "__main__":
    main()
