"""Compute the numbers for the grouped-query attention example.

Run: python3 scripts/gqa_example.py
"""

import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(Q, K, V):
    return softmax(Q @ K.T / np.sqrt(Q.shape[-1])) @ V


def gqa(X, Wq, Wk, Wv, Wo, H, G):
    """Head i (0-based) reads key/value group i // (H // G)."""
    n = X.shape[0]
    d_k = Wq.shape[1] // H
    r = H // G
    outs = []
    for i in range(H):
        g = i // r
        q = X @ Wq[:, i * d_k:(i + 1) * d_k]
        k = X @ Wk[:, g * d_k:(g + 1) * d_k]
        v = X @ Wv[:, g * d_k:(g + 1) * d_k]
        outs.append(attention(q, k, v))
    return np.concatenate(outs, axis=-1) @ Wo


H, d_k, d_model, n = 4, 2, 8, 3
print(f"H = {H}, d_k = {d_k}, d_model = {d_model}")
for G in (4, 2, 1):
    r = H // G
    groups = [i // r for i in range(H)]
    print(f"G = {G}: query head -> group {groups}, K/V heads cached = {G}, K/V reduction = {H // G}x")
print("interleaved assignment for G = 2 (a different choice):", [i % 2 for i in range(H)])

rng = np.random.default_rng(0)
X = rng.normal(size=(n, d_model))
Wq = rng.normal(size=(d_model, H * d_k)) / np.sqrt(d_model)
Wo = rng.normal(size=(H * d_k, d_model)) / np.sqrt(H * d_k)
Wk_full = rng.normal(size=(d_model, H * d_k)) / np.sqrt(d_model)
Wv_full = rng.normal(size=(d_model, H * d_k)) / np.sqrt(d_model)

# G = H: independent K/V per head, which is multi-head attention.
mha = gqa(X, Wq, Wk_full, Wv_full, Wo, H, H)
# Reference: the plain multi-head formula, head by head.
ref = np.concatenate(
    [attention(X @ Wq[:, i * d_k:(i + 1) * d_k], X @ Wk_full[:, i * d_k:(i + 1) * d_k], X @ Wv_full[:, i * d_k:(i + 1) * d_k]) for i in range(H)],
    axis=-1,
) @ Wo
print()
print("GQA with G = H matches multi-head attention:", bool(np.allclose(mha, ref)))

# G = 1: every head reads the same K and V, which is multi-query attention.
Wk1, Wv1 = Wk_full[:, :d_k], Wv_full[:, :d_k]
mqa = gqa(X, Wq, Wk1, Wv1, Wo, H, 1)
print("GQA with G = 1 reads one shared K/V head; output row 0 =", np.round(mqa[0], 4).tolist())
print("MHA output row 0                                    =", np.round(mha[0], 4).tolist())

# Cached floats per token per layer.
for G in (4, 2, 1):
    print(f"G = {G}: floats cached per token per layer = 2 * {G} * {d_k} = {2 * G * d_k}")
