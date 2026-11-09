"""Compute the numbers for the LoRA forward pass, merge, and gradient examples.

Run: python3 scripts/lora_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Paper shapes (Section 4.1): W0 is d x k, B is d x r, A is r x k.
W0 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0]])
B = np.array([[1.0], [2.0]])
A = np.array([[1.0, 0.0, -1.0]])
x = np.array([1.0, 2.0, 3.0])
d, k = W0.shape
r = A.shape[0]

print("W0 x            =", (W0 @ x).tolist())
print("A x             =", (A @ x).tolist())
print("B (A x)         =", (B @ (A @ x)).tolist())
print("h = W0x + BAx   =", (W0 @ x + B @ (A @ x)).tolist())
print("Delta W = B A   =", (B @ A).tolist())
print("rank(Delta W)   =", np.linalg.matrix_rank(B @ A))
W = W0 + B @ A
print("merged W        =", W.tolist())
print("merged W x      =", (W @ x).tolist())

# Scaling by alpha / r (Section 4.1). Here r = 1, so alpha = 2 doubles the update.
alpha = 2.0
s = alpha / r
print("alpha/r         =", s)
print("h with alpha=2  =", (W0 @ x + s * B @ (A @ x)).tolist())

# Zero B at the start (Section 4.1): the layer is the base layer.
B0 = np.zeros((d, r))
print("h with B = 0    =", (W0 @ x + B0 @ (A @ x)).tolist())

# Gradients at the zero initialization. Upstream gradient dh = dL/dh.
dh = np.array([1.0, 1.0])
for name, Bm in [("B = 0", B0), ("B = [1, 2]^T", B)]:
    dB = s * np.outer(dh, A @ x)
    dA = s * np.outer(Bm.T @ dh, x)
    print(f"{name:13s} dB =", dB.tolist(), " dA =", dA.tolist())

# Unmerge and swap (Section 4.1, "No Additional Inference Latency").
B2 = np.array([[0.0], [1.0]])
A2 = np.array([[2.0, 0.0, 0.0]])
W_swapped = W - B @ A + B2 @ A2
print("W - BA          =", (W - B @ A).tolist())
print("swapped W       =", W_swapped.tolist())
print("recovered W0    =", bool(np.allclose(W - B @ A, W0)))
