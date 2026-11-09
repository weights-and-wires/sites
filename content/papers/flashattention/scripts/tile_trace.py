#!/usr/bin/env python3
"""Trace Algorithm 1 of the FlashAttention paper on a tiny example.

Prints the running statistics m and l and the running output O for one row
after each K/V block, then compares the final result with naive attention.
Also shows that naive softmax overflows on large scores and the tiled
algorithm doesn't.

Run: python3 content/papers/flashattention/scripts/tile_trace.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

Q = np.array([[1.0, 0.0], [0.5, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
Bc = 2


def naive(Q, K, V):
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V


def tiled(Q, K, V, Bc, verbose=False, row=0):
    """Algorithm 1 with one Q block (Br = N). Follows lines 5-13."""
    N, d = Q.shape
    O = np.zeros((N, d), dtype=Q.dtype)
    l = np.zeros(N, dtype=Q.dtype)
    m = np.full(N, -np.inf, dtype=Q.dtype)
    for j, start in enumerate(range(0, K.shape[0], Bc), start=1):
        Kj, Vj = K[start:start + Bc], V[start:start + Bc]
        S = Q @ Kj.T                                    # line 9
        m_t = S.max(axis=1)                             # line 10
        P_t = np.exp(S - m_t[:, None])
        l_t = P_t.sum(axis=1)
        m_new = np.maximum(m, m_t)                      # line 11
        l_new = np.exp(m - m_new) * l + np.exp(m_t - m_new) * l_t
        O = ((l * np.exp(m - m_new))[:, None] * O       # line 12
             + np.exp(m_t - m_new)[:, None] * (P_t @ Vj)) / l_new[:, None]
        l, m = l_new, m_new                             # line 13
        if verbose:
            print(f"block {j}: S row {row} = {S[row]}, m~ = {m_t[row]:.4f}, l~ = {l_t[row]:.4f}")
            print(f"         m = {m[row]:.4f}, l = {l[row]:.4f}, O row {row} = {O[row]}")
    return O, m, l


print("Q row 0 =", Q[0], "; K rows =", K.tolist(), "; Bc =", Bc)
O, m, l = tiled(Q, K, V, Bc, verbose=True)
want = naive(Q, K, V)
print("tiled O =", O.tolist())
print("naive O =", want.tolist())
print("max |difference| =", float(np.abs(O - want).max()))
S = Q @ K.T
print("full row 0 scores =", S[0], "; row max =", S[0].max(), "; row sum of exp(S - max) =", round(float(np.exp(S[0] - S[0].max()).sum()), 4))

print()
print("Large scores (float32):")
Q32 = np.array([[1000.0, 0.0]], dtype=np.float32)
K32 = np.array([[1.0, 0.0], [0.9, 0.0], [0.0, 1.0], [0.8, 0.0]], dtype=np.float32)
V32 = np.array([[1, 0], [0, 1], [1, 1], [2, 2]], dtype=np.float32)
with np.errstate(over="ignore", invalid="ignore"):
    E = np.exp(Q32 @ K32.T)
    naive_out = (E / E.sum(axis=1, keepdims=True)) @ V32
print("scores =", Q32 @ K32.T)
print("softmax without the max shift:", naive_out)
out, m32, l32 = tiled(Q32, K32, V32, 2)
print("tiled output:", out, "; m =", m32, "; l =", l32)
