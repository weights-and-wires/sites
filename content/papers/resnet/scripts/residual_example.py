"""Compute the numbers for the residual block examples.

Run: python3 scripts/residual_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def relu(x):
    return np.maximum(x, 0.0)


def block(x, W1, W2, Ws=None):
    F = W2 @ relu(W1 @ x)
    return F, relu(F + (x if Ws is None else Ws @ x))


# Worked block (Equation 1)
W1 = np.eye(2)
W2 = np.array([[1.0, -1.0], [0.5, 0.5]])
x = np.array([2.0, -1.0])
F, y = block(x, W1, W2)
print("Worked block, identity shortcut (Equation 1)")
print("x                =", x.tolist())
print("relu(W1 x)       =", relu(W1 @ x).tolist())
print("F(x) = W2 relu(W1 x) =", F.tolist())
print("F(x) + x         =", (F + x).tolist())
print("y = relu(F + x)  =", y.tolist())

# Constructed identity solution: the added block has W1 = W2 = 0
print()
print("Added block with zero weights, input x >= 0")
xp = np.array([0.5, 2.0, 7.0])
Z = np.zeros((3, 3))
_, y_res = block(xp, Z, Z)
y_plain = relu(Z @ xp)
print("x                =", xp.tolist())
print("residual block   =", y_res.tolist(), " (identity)")
print("plain layer      =", y_plain.tolist(), " (zero; it must learn W = I)")
print("plain layer with W = I =", relu(np.eye(3) @ xp).tolist())

# Projection shortcut (Equation 2)
print()
print("Projection shortcut (Equation 2), 2 -> 3 dimensions")
W1p, W2p = np.ones((4, 2)), np.ones((3, 4))
Ws = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
xq = np.array([1.0, 2.0])
Fq, yq = block(xq, W1p, W2p, Ws)
print("x                =", xq.tolist())
print("F(x)             =", Fq.tolist())
print("Ws x             =", (Ws @ xq).tolist())
print("y                =", yq.tolist())
