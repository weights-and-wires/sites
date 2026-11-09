"""Compute the numbers for the multi-head attention example.

Run: python3 scripts/multihead_example.py
"""

import numpy as np

# Parameter counts for the base model (Section 3.2.2: h = 8, d_model = 512).
d_model, h = 512, 8
d_k = d_v = d_model // h
per_head = 3 * d_model * d_k  # W_i^Q, W_i^K, W_i^V
heads = h * per_head
w_o = (h * d_v) * d_model
print(f"d_k = d_v = {d_k}")
print(f"one head: W^Q, W^K, W^V = 3 * {d_model} * {d_k} = {per_head}")
print(f"{h} heads: {heads}; W^O: {w_o}; total {heads + w_o}")
print(f"one head of width d_model: 4 * {d_model}^2 = {4 * d_model * d_model}")

# Shape walk-through with n = 3 tokens, d_model = 4, h = 2.
n, d, hh = 3, 4, 2
dk = d // hh
X = np.arange(n * d, dtype=float).reshape(n, d)
print()
print("X (n=3, d_model=4):")
print(X.astype(int).tolist())
split = X.reshape(n, hh, dk).transpose(1, 0, 2)  # (h, n, d_k)
print("shape after reshape + transpose:", split.shape)
print("head 0 sees columns 0-1:", split[0].astype(int).tolist())
print("head 1 sees columns 2-3:", split[1].astype(int).tolist())
back = split.transpose(1, 0, 2).reshape(n, d)
print("concat restores X:", bool(np.array_equal(back, X)))
wrong = X.reshape(hh, n, dk)
print("reshape without transpose, head 0:", wrong[0].astype(int).tolist(), "(mixes tokens)")
