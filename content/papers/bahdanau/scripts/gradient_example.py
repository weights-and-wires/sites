"""Compute the numbers for the soft-alignment gradient example.

Run: python3 scripts/gradient_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

e = np.array([1.0, 0.0, -1.0])
H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
g = np.array([1.0, 0.0])  # dL/dc, so L = g . c

alpha = np.exp(e) / np.exp(e).sum()
c = alpha @ H
dalpha = H @ g
de = alpha * (dalpha - alpha @ dalpha)

print("alpha:", alpha)
print("c:", c)
print("dL/dalpha = H g:", dalpha)
print("alpha . dalpha:", alpha @ dalpha)
print("dL/de:", de, " sum =", de.sum())

# Finite-difference check of dL/de.
eps = 1e-6
fd = np.zeros(3)
for k in range(3):
    for sign in (+1, -1):
        ee = e.copy()
        ee[k] += sign * eps
        a = np.exp(ee) / np.exp(ee).sum()
        fd[k] += sign * (a @ H @ g) / (2 * eps)
print("finite differences:", fd)
