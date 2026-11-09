"""Compute the numbers for the Megatron-LM MLP-split examples.

Run: python3 scripts/mlp_split.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def gelu(x):
    """The tanh approximation of GeLU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# Row split (Equation 2): GeLU(X1 A1 + X2 A2) != GeLU(X1 A1) + GeLU(X2 A2).
X = np.array([[1.0, 2.0]])
A = np.array([[3.0], [-2.0]])
p1, p2 = X[:, :1] @ A[:1], X[:, 1:] @ A[1:]
print("row split: X1 A1 =", p1.ravel(), " X2 A2 =", p2.ravel(), " sum =", (p1 + p2).ravel())
print("  GeLU(sum)            =", gelu(p1 + p2).ravel())
print("  GeLU(p1) + GeLU(p2)  =", (gelu(p1) + gelu(p2)).ravel())

# Column split (Equation 3), then a row-split second GEMM, on 2 GPUs.
X = np.array([[1.0, 2.0], [0.0, -1.0]])  # 2 tokens, h = 2
A = np.array([[1.0, -1.0, 0.5, 2.0], [0.0, 1.0, -2.0, 1.0]])  # h = 2, width 4
B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]])  # 4 x 2
full = gelu(X @ A) @ B
A1, A2 = A[:, :2], A[:, 2:]
B1, B2 = B[:2], B[2:]
Z1 = gelu(X @ A1) @ B1
Z2 = gelu(X @ A2) @ B2
print()
print("single-device output Z =", full.round(4).tolist())
print("GPU 1 partial Z1       =", Z1.round(4).tolist())
print("GPU 2 partial Z2       =", Z2.round(4).tolist())
print("all-reduce Z1 + Z2     =", (Z1 + Z2).round(4).tolist())
print("max |Z1 + Z2 - Z|      =", np.abs(Z1 + Z2 - full).max())

# 4-way split on a larger random MLP: how close is the sum to the single-device result?
rng = np.random.default_rng(0)
h, t = 8, 4
X = rng.normal(size=(5, h))
A = rng.normal(size=(h, 4 * h))
B = rng.normal(size=(4 * h, h))
full = gelu(X @ A) @ B
parts = [gelu(X @ a) @ b for a, b in zip(np.split(A, t, axis=1), np.split(B, t, axis=0))]
print()
print(f"4-way, h = {h}: shard shapes A_i {np.split(A, t, axis=1)[0].shape}, B_i {np.split(B, t, axis=0)[0].shape}")
print("max |sum of partials - single device| =", f"{np.abs(sum(parts) - full).max():.2e}")
print("max |one partial - single device|     =", f"{np.abs(parts[0] - full).max():.4f}")
