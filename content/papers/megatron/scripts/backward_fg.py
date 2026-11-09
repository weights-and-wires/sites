"""Compute the numbers for the f and g operator example (Figure 3a, Code 1).

Run: python3 scripts/backward_fg.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# One linear layer Y = X A with A split by columns across 2 GPUs: Y_i = X A_i.
# The loss gradient arrives as dY = [dY_1, dY_2].
X = np.array([[1.0, 2.0]])
A = np.array([[1.0, 0.0, 2.0, -1.0], [3.0, 1.0, 0.0, 1.0]])
dY = np.array([[1.0, 2.0, -1.0, 0.5]])
A1, A2 = A[:, :2], A[:, 2:]
dY1, dY2 = dY[:, :2], dY[:, 2:]

dX_full = dY @ A.T
dX1 = dY1 @ A1.T
dX2 = dY2 @ A2.T
print("single device dX = dY A^T      =", dX_full.tolist())
print("GPU 1 local gradient dY_1 A_1^T =", dX1.tolist())
print("GPU 2 local gradient dY_2 A_2^T =", dX2.tolist())
print("all-reduce (f operator, backward) =", (dX1 + dX2).tolist())

# The weight gradients need no communication: dA_i = X^T dY_i.
dA_full = X.T @ dY
print()
print("dA_1 = X^T dY_1 =", (X.T @ dY1).tolist())
print("dA_2 = X^T dY_2 =", (X.T @ dY2).tolist())
print("concatenated equals single-device dA:", np.allclose(np.hstack([X.T @ dY1, X.T @ dY2]), dA_full))
