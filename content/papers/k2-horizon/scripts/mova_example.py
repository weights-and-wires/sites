"""Mixture-of-Value Attention (Section 2.2, Equations 5-6) on one token.

The router normalization (softmax over all experts, then renormalize over the
top k) is this page's reading; the paper's Equation 6 only names p_i(x_t).
"""
import numpy as np


def silu(z):
    return z / (1 + np.exp(-z))


x = np.array([1.0, -1.0])
# Four value experts, each a 2x2 matrix. Router weights are 2x4.
W_v = np.array([
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
    [[2.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, -2.0]],
])
W_r = np.array([[1.0, 0.0, 2.0, 0.5], [0.0, 1.0, -1.0, 0.0]])
k = 2

logits = x @ W_r
idx = np.argsort(-logits)[:k]
p = np.exp(logits[idx] - logits[idx].max())
p = p / p.sum()
v = sum(p_i * silu(x @ W_v[i]) for p_i, i in zip(p, idx))
np.set_printoptions(precision=4, suppress=True)
print("router logits", logits, "-> top-2 experts", idx.tolist(), "weights", p)
for i in idx:
    print(f"expert {i}: x W = {x @ W_v[i]}, SiLU = {silu(x @ W_v[i])}")
print("value v(x) =", v)

# KV cache per token per layer: MoVA caches one mixed value per KV head, as GQA does.
kv_heads, head_dim = 8, 128  # head_dim = 128 is an assumption, not from the paper
print("\nKV elements per token per layer, GQA or MoVA:", 2 * kv_heads * head_dim)
