"""Compute the numbers for the scaled dot-product attention and masking examples.

Run: python3 scripts/attention_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [1.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0]])
d_k = Q.shape[-1]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


scores = Q @ K.T
scaled = scores / np.sqrt(d_k)
weights = softmax(scaled)
out = weights @ V
print("Q K^T            =", scores.tolist())
print("Q K^T / sqrt(d_k)=", scaled.tolist())
print("softmax rows     =", weights.tolist())
print("output           =", out.tolist())
print("row 2 by hand    =", (weights[1, 0] * V[0] + weights[1, 1] * V[1]).tolist())

print()
print("Causal mask example (same Q, K, V)")
mask = np.tril(np.ones((2, 2), dtype=bool))
masked = np.where(mask, scaled, -1e9)
w_masked = softmax(masked)
print("masked scores    =", masked.tolist())
print("masked weights   =", w_masked.tolist())
print("masked output    =", (w_masked @ V).tolist())

print()
print("Wrong: mask with 0 instead of -1e9")
w_zero = softmax(np.where(mask, scaled, 0.0))
print("weights          =", w_zero.tolist())
