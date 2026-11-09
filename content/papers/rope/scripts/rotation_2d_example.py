"""Compute the numbers for the 2D rotation example (Section 3.2.1, Equations 12 and 13).

Run: python3 scripts/rotation_2d_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

theta = 0.5
q = np.array([1.0, 2.0])  # W_q x_m
k = np.array([3.0, 1.0])  # W_k x_n


def R(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


print("q . k with no rotation:", q @ k)
print()
print(" m  n  m-n   q_m . k_n")
for m, n in [(5, 2), (8, 5), (13, 10), (2, 5), (7, 2)]:
    qm, kn = R(m * theta) @ q, R(n * theta) @ k
    print(f"{m:2d} {n:2d} {m - n:4d}   {qm @ kn:.4f}")

# Equation 12: the same score as the real part of a product of complex numbers.
qc, kc = complex(*q), complex(*k)
m, n = 5, 2
g = (qc * np.exp(1j * m * theta) * np.conj(kc * np.exp(1j * n * theta))).real
print()
print(f"complex form for m=5, n=2: {g:.4f}")
print(f"Re[q k* e^(i(m-n)theta)]:  {(qc * np.conj(kc) * np.exp(1j * (m - n) * theta)).real:.4f}")

print()
print("norms at m=5:", np.linalg.norm(q), np.linalg.norm(R(5 * theta) @ q))
print("R(3 theta) @ [1, 0] =", R(3 * theta) @ np.array([1.0, 0.0]))
