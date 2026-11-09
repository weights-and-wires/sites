"""Compute the numbers for the general-form example (Section 3.2.2, Equations 14 to 16, 34).

Run: python3 scripts/rope_full_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def thetas(d, base=10000.0):
    # Equation 15: theta_i = base^(-2(i-1)/d) for i = 1..d/2.
    return base ** (-2.0 * np.arange(d // 2) / d)


def rotary_matrix(m, d, base=10000.0):
    R = np.zeros((d, d))
    for i, th in enumerate(thetas(d, base)):
        c, s = np.cos(m * th), np.sin(m * th)
        R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, -s], [s, c]]
    return R


def rope(x, m, base=10000.0):
    # Equation 34.
    d = len(x)
    th = thetas(d, base)
    cos, sin = np.repeat(np.cos(m * th), 2), np.repeat(np.sin(m * th), 2)
    x_swapped = np.empty_like(x)
    x_swapped[0::2], x_swapped[1::2] = -x[1::2], x[0::2]
    return x * cos + x_swapped * sin


d = 8
print("d = 8 thetas:", thetas(d).tolist())
print("wavelengths 2*pi/theta:", (2 * np.pi / thetas(d)).tolist())
print("d = 128, last theta:", thetas(128)[-1], " wavelength:", 2 * np.pi / thetas(128)[-1])

x = np.arange(1.0, 9.0)
m = 3
print()
print("x      =", x)
print("R_3 x  =", rotary_matrix(m, d) @ x)
print("eq 34  =", rope(x, m))
print("pair 1 angle:", m * thetas(d)[0], " pair 4 angle:", m * thetas(d)[3])

rng = np.random.default_rng(0)
q, k = rng.normal(size=d), rng.normal(size=d)
print()
print("m   n   m-n   q_m . k_n")
for m, n in [(0, 0), (4, 4), (9, 9), (3, 1), (10, 8), (50, 48), (1, 4), (6, 9), (100, 103)]:
    print(f"{m:3d} {n:3d} {m - n:4d}   {rope(q, m) @ rope(k, n):.6f}")

m, n = 7, 3
lhs = rope(q, m) @ rope(k, n)
rhs = q @ rotary_matrix(n - m, d) @ k
print()
print(f"Equation 16 check (m=7, n=3): {lhs:.6f} vs q^T R_(n-m) k = {rhs:.6f}")
print("R_m^T R_n == R_(n-m):", np.allclose(rotary_matrix(m, d).T @ rotary_matrix(n, d), rotary_matrix(n - m, d)))
print("orthogonal:", np.allclose(rotary_matrix(m, d).T @ rotary_matrix(m, d), np.eye(d)))
