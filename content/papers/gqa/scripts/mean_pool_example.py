"""Compute the numbers for the mean-pooling (uptraining) example.

Run: python3 scripts/mean_pool_example.py
"""

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def mean_pool(W, H, G):
    """Average the column blocks (heads) of W within each of G contiguous groups."""
    d_model = W.shape[0]
    d_k = W.shape[1] // H
    return W.reshape(d_model, G, H // G, d_k).mean(axis=2).reshape(d_model, G * d_k)


# Hand-checkable case: d_model = 2, H = 4 heads of width d_k = 1.
W = np.array([[1, 3, 5, 7], [2, 4, 6, 8]], dtype=float)
print("W^K, columns are heads 0-3:", W.astype(int).tolist())
for G in (2, 1):
    print(f"mean-pooled to G = {G}:", mean_pool(W, 4, G).tolist())
x = np.array([1.0, 1.0])
print("keys for x = [1, 1] before pooling:", (x @ W).tolist())
print("keys after pooling to G = 2:", (x @ mean_pool(W, 4, 2)).tolist())

# Conversion changes the function unless the heads already agree.
rng = np.random.default_rng(0)
H, G, d_k, d_model, n = 4, 2, 2, 8, 5
X = rng.normal(size=(n, d_model))
Q = X @ (rng.normal(size=(d_model, H * d_k)) / np.sqrt(d_model))
Wk = rng.normal(size=(d_model, H * d_k)) / np.sqrt(d_model)
Wv = rng.normal(size=(d_model, H * d_k)) / np.sqrt(d_model)


def run(Wk, Wv, H, G):
    r = H // G
    outs = []
    for i in range(H):
        g = i // r
        q = Q[:, i * d_k:(i + 1) * d_k]
        k = X @ Wk[:, g * d_k:(g + 1) * d_k]
        v = X @ Wv[:, g * d_k:(g + 1) * d_k]
        outs.append(softmax(q @ k.T / np.sqrt(d_k)) @ v)
    return np.concatenate(outs, axis=-1)


before = run(Wk, Wv, H, H)
after = run(mean_pool(Wk, H, G), mean_pool(Wv, H, G), H, G)
print()
print(f"random heads: max |MHA - converted GQA-{G}| = {np.abs(before - after).max():.4f}")

# Heads that already share weights convert exactly.
blocks = lambda B: np.concatenate([B[:, :d_k]] * 2 + [B[:, d_k:2 * d_k]] * 2, axis=1)
Wk_s, Wv_s = blocks(Wk), blocks(Wv)
before = run(Wk_s, Wv_s, H, H)
after = run(mean_pool(Wk_s, H, G), mean_pool(Wv_s, H, G), H, G)
print(f"heads that already match within each group: max error = {np.abs(before - after).max():.1e}")
