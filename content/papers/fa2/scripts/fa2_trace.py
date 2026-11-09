#!/usr/bin/env python3
"""Trace FlashAttention-2 Algorithm 1 on a tiny example.

Prints the unscaled output, the running statistics, and the logsumexp for one
query row, then compares the loop order of FlashAttention (K/V outer) with
FlashAttention-2 (Q outer) on random data, and counts HBM traffic for both.

Run: python3 content/papers/fa2/scripts/fa2_trace.py
"""

import math

import numpy as np

np.set_printoptions(precision=4, suppress=True)

q = np.array([[1.0, 0.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [1.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
Bc = 2

print("scores:", (q @ K.T)[0])
m = np.array([-np.inf])
l = np.zeros(1)
O_t = np.zeros((1, 2))
for j in range(0, 4, Bc):
    S = q @ K[j:j + Bc].T
    m_new = np.maximum(m, S.max(axis=1))
    P = np.exp(S - m_new[:, None])
    alpha = np.exp(m - m_new)
    l = alpha * l + P.sum(axis=1)
    O_t = alpha[:, None] * O_t + P @ V[j:j + Bc]
    m = m_new
    print(f"after block {j // Bc + 1}: alpha={alpha[0]:.4f} m={m[0]:.4f} l={l[0]:.4f} "
          f"O_tilde={O_t[0]}  (O_tilde / l = {(O_t / l[:, None])[0]})")
O = O_t / l[:, None]
L = m + np.log(l)
print("final O:", O[0], " L =", f"{L[0]:.4f}")

S = q @ K.T
want = (np.exp(S - S.max()) / np.exp(S - S.max()).sum()) @ V
print("naive O:", want[0], " naive L =", f"{np.log(np.exp(S).sum()):.4f}")
print("max difference:", f"{np.abs(O - want).max():.1e}")

# L equals m + log(l) for FA1's two statistics.
print("m + log(l) from the two FA1 statistics:", f"{m[0] + math.log(l[0]):.4f}")


def tiled_kv_outer(Q, K, V, Br, Bc):
    """FlashAttention loop order (K/V outer), normalized output every step."""
    N = Q.shape[0]
    O = np.zeros((N, V.shape[1]))
    ll = np.zeros(N)
    mm = np.full(N, -np.inf)
    for j in range(0, K.shape[0], Bc):
        for i in range(0, N, Br):
            S = Q[i:i + Br] @ K[j:j + Bc].T
            mt = S.max(axis=1)
            Pt = np.exp(S - mt[:, None])
            lt = Pt.sum(axis=1)
            mn = np.maximum(mm[i:i + Br], mt)
            ln = np.exp(mm[i:i + Br] - mn) * ll[i:i + Br] + np.exp(mt - mn) * lt
            O[i:i + Br] = ((ll[i:i + Br] * np.exp(mm[i:i + Br] - mn))[:, None] * O[i:i + Br]
                           + (np.exp(mt - mn)[:, None] * (Pt @ V[j:j + Bc]))) / ln[:, None]
            ll[i:i + Br], mm[i:i + Br] = ln, mn
    return O


def tiled_q_outer(Q, K, V, Br, Bc):
    """FlashAttention-2 loop order (Q outer), unscaled output."""
    N = Q.shape[0]
    O = np.zeros((N, V.shape[1]))
    for i in range(0, N, Br):
        Ot = np.zeros((min(Br, N - i), V.shape[1]))
        ll = np.zeros(Ot.shape[0])
        mm = np.full(Ot.shape[0], -np.inf)
        for j in range(0, K.shape[0], Bc):
            S = Q[i:i + Br] @ K[j:j + Bc].T
            mn = np.maximum(mm, S.max(axis=1))
            P = np.exp(S - mn[:, None])
            a = np.exp(mm - mn)
            ll = a * ll + P.sum(axis=1)
            Ot = a[:, None] * Ot + P @ V[j:j + Bc]
            mm = mn
        O[i:i + Br] = Ot / ll[:, None]
    return O


rng = np.random.default_rng(0)
Q, K2, V2 = rng.normal(size=(64, 8)), rng.normal(size=(64, 8)), rng.normal(size=(64, 8))
a, b = tiled_kv_outer(Q, K2, V2, 16, 16), tiled_q_outer(Q, K2, V2, 16, 16)
print("\nloop order, N=64 d=8 blocks 16x16: max |FA1 order - FA2 order| =", f"{np.abs(a - b).max():.1e}")

# HBM traffic in elements, one head, N=4096, d=64, Br=Bc=128.
N, d, Br, Bc = 4096, 64, 128, 128
Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
fa1 = 2 * N * d + Tc * ((N * d + N * d + N + N) + (N * d + N + N))  # K,V once; Q,O,l,m in and O,l,m out per K/V block
fa2 = N * d + Tr * (2 * N * d) + N * d + N  # Q once; K,V per Q block; O and L out once
print(f"\nHBM elements, N={N} d={d} Br=Bc={Br}: Tr={Tr}, Tc={Tc}")
print(f"FA1: {fa1:,}   FA2: {fa2:,}   ratio FA1/FA2 = {fa1 / fa2:.2f}")
