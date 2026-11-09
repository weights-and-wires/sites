#!/usr/bin/env python3
"""The Cholesky form of H^-1 (Step 3) reproduces the rows of Equation 3's elimination."""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

H = np.array([[4.0, 2.0, 1.0, 0.0],
              [2.0, 6.0, 2.0, 1.0],
              [1.0, 2.0, 6.0, 2.0],
              [0.0, 1.0, 2.0, 5.0]])
Hinv = np.linalg.inv(H)
U = np.linalg.cholesky(Hinv).T
print("U = Cholesky(H^-1)^T =")
print(U)
print(f"max |U^T U - H^-1| = {np.abs(U.T @ U - Hinv).max():.2e}")

# Equation 3, one column at a time: remove row and column q from the inverse.
cur = Hinv.copy()
for j in range(4):
    row = cur[0, :]  # row of the current inverse for column j, restricted to the remaining columns
    want = row / np.sqrt(row[0])
    got = U[j, j:]
    print(f"j={j}: [H^-1_F]_jj = {row[0]:.4f}, U_jj^2 = {U[j, j] ** 2:.4f}, "
          f"max |row/sqrt(diag) - U[j, j:]| = {np.abs(want - got).max():.2e}")
    cur = cur - np.outer(cur[:, 0], cur[0, :]) / cur[0, 0]
    cur = cur[1:, 1:]
