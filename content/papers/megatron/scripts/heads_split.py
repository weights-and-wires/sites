"""Compute the numbers for the attention-head split example (Figure 3b).

Run: python3 scripts/heads_split.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def heads_attention(X, Wq, Wk, Wv, heads):
    """Attention for the heads held in these column slices; returns (n, heads * d_k)."""
    n = X.shape[0]
    d_k = Wq.shape[1] // heads
    split = lambda M: M.reshape(n, heads, d_k).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq), split(X @ Wk), split(X @ Wv)
    w = softmax(Q @ K.transpose(0, 2, 1) / np.sqrt(d_k))
    return (w @ V).transpose(1, 0, 2).reshape(n, heads * d_k)


rng = np.random.default_rng(1)
n, d, heads, t = 6, 16, 4, 2
d_k = d // heads
X = rng.normal(size=(n, d))
Wq, Wk, Wv, Wo = (rng.normal(size=(d, d)) * 0.3 for _ in range(4))

full = heads_attention(X, Wq, Wk, Wv, heads) @ Wo
cols = d // t
parts = []
for r in range(t):
    s = slice(r * cols, (r + 1) * cols)
    parts.append(heads_attention(X, Wq[:, s], Wk[:, s], Wv[:, s], heads // t) @ Wo[s])

print(f"d = {d}, heads = {heads}, d_k = {d_k}, GPUs = {t}")
print(f"GPU r holds {heads // t} heads: columns {[(r * cols, (r + 1) * cols) for r in range(t)]} of Wq, Wk, Wv")
print(f"per-GPU shard shapes: Wq_r {Wq[:, :cols].shape}, Wo_r {Wo[:cols].shape}")
print("max |sum of partials - single device| =", f"{np.abs(sum(parts) - full).max():.2e}")

# Mixing up scale: each GPU divides by sqrt of its local width instead of sqrt(d_k).
print()
print(f"correct scale 1/sqrt(d_k) = 1/sqrt({d_k}) = {1 / np.sqrt(d_k):.4f}")
print(f"wrong scale 1/sqrt(local width) = 1/sqrt({cols}) = {1 / np.sqrt(cols):.4f}")

# Parameters per GPU for the paper's 8.3B model: attention has 4 h^2 weights.
h = 3072
print()
print(f"attention weights per layer (h = {h}): {4 * h * h:,}; per GPU with t = 8: {4 * h * h // 8:,}")
