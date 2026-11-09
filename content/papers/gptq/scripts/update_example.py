#!/usr/bin/env python3
"""One error-compensation step (Equation 2) on a 4 x 4 Hessian, checked against a direct solve."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

H = np.array([[4.0, 2.0, 1.0, 0.0],
              [2.0, 6.0, 2.0, 1.0],
              [1.0, 2.0, 6.0, 2.0],
              [0.0, 1.0, 2.0, 5.0]])
w = np.array([0.37, -0.82, 0.55, 0.14])
step = 0.25
q = 0

Hinv = np.linalg.inv(H)
wq = np.round(w[q] / step) * step
err = w[q] - wq
delta = -(w[q] - wq) / Hinv[q, q] * Hinv[:, q]
w_new = w + delta

print("H^-1 =")
print(Hinv)
print(f"quantize w[0] = {w[0]} to {wq}; error {err:.4f}")
print(f"[H^-1]_00 = {Hinv[q, q]:.4f}")
print("delta =", delta)
print("w + delta =", w_new)

# Direct solve: minimize d^T H d / 2 over the perturbation d with d[0] = wq - w[0].
free = [1, 2, 3]
e = wq - w[q]
d_free = -np.linalg.solve(H[np.ix_(free, free)], H[free, q] * e)
d = np.zeros(4)
d[q], d[free] = e, d_free
print("direct solve w + d =", w + d)
print(f"max |Equation 2 - direct solve| = {np.abs(delta - d).max():.2e}")

# Loss increase for Equation 1 is d X X^T d^T = d H d^T / 2.
loss_eq2 = 0.5 * delta @ H @ delta
loss_naive = 0.5 * (e**2) * H[q, q]
print(f"loss increase with compensation    {loss_eq2:.5f}")
print(f"loss increase without compensation {loss_naive:.5f}")
print(f"e^2 / [H^-1]_00 / 2                {0.5 * e**2 / Hinv[q, q]:.5f}")

# Unconstrained weights stay put when they do not interact with column 0.
print("components of delta:", delta)
