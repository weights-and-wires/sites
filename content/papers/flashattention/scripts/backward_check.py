#!/usr/bin/env python3
"""Check the recomputation-based backward pass of FlashAttention (Appendix B.2, B.4).

Part 1 works one softmax row by hand: D_i = dO_i . O_i equals the sum of
P_ij dP_ij (Appendix B.2, Equation 4).
Part 2 runs Algorithm 4 without dropout and masking, from only Q, K, V, O,
dO, m, and l, and compares it with the gradients of naive attention.

Run: python3 content/papers/flashattention/scripts/backward_check.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Part 1: one query row, two keys.
q = np.array([1.0, 0.0])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[1.0, 2.0], [3.0, 4.0]])
dO = np.array([1.0, 0.0])
s = K @ q
P = np.exp(s - s.max())
P /= P.sum()
o = P @ V
dP = V @ dO                      # dP_j = dO . v_j
D_direct = float(P @ dP)         # sum_j P_j dP_j
D_shortcut = float(dO @ o)       # Equation 4
dS = P * (dP - D_shortcut)
print("scores s =", s)
print("P =", P)
print("o =", o)
print("dP =", dP)
print(f"D = sum_j P_j dP_j = {D_direct:.4f}; D = dO . o = {D_shortcut:.4f}")
print("dS = P * (dP - D) =", dS, "; sum =", round(float(dS.sum()), 12))


# Part 2: full algorithm.
def forward(Q, K, V, Bc):
    N, d = Q.shape
    O, l, m = np.zeros((N, d)), np.zeros(N), np.full(N, -np.inf)
    for j in range(0, N, Bc):
        S = Q @ K[j:j + Bc].T
        m_t = S.max(axis=1)
        P_t = np.exp(S - m_t[:, None])
        l_t = P_t.sum(axis=1)
        m_new = np.maximum(m, m_t)
        l_new = np.exp(m - m_new) * l + np.exp(m_t - m_new) * l_t
        O = ((l * np.exp(m - m_new))[:, None] * O + np.exp(m_t - m_new)[:, None] * (P_t @ V[j:j + Bc])) / l_new[:, None]
        l, m = l_new, m_new
    return O, m, l


def backward(Q, K, V, O, dO, m, l, Br, Bc):
    """Algorithm 4 with tau = 1, no mask, no dropout."""
    N, d = Q.shape
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for j in range(0, N, Bc):
        Kj, Vj = K[j:j + Bc], V[j:j + Bc]
        dKj, dVj = np.zeros_like(Kj), np.zeros_like(Vj)
        for i in range(0, N, Br):
            Qi, Oi, dOi = Q[i:i + Br], O[i:i + Br], dO[i:i + Br]
            S = Qi @ Kj.T                                    # line 11
            Pij = np.exp(S - m[i:i + Br, None]) / l[i:i + Br, None]   # line 13
            dVj += Pij.T @ dOi                               # line 16
            dPij = dOi @ Vj.T                                # line 17
            Di = (dOi * Oi).sum(axis=1)                      # line 19
            dS = Pij * (dPij - Di[:, None])                  # line 20
            dQ[i:i + Br] += dS @ Kj                          # line 21
            dKj += dS.T @ Qi                                 # line 22
        dK[j:j + Bc], dV[j:j + Bc] = dKj, dVj
    return dQ, dK, dV


def naive_backward(Q, K, V, dO):
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - (P * dP).sum(axis=1, keepdims=True))
    return dS @ K, dS.T @ Q, dV


rng = np.random.default_rng(0)
N, d, Br, Bc = 10, 4, 3, 4
Q, K, V, dOm = (rng.normal(size=(N, d)) for _ in range(4))
O, m, l = forward(Q, K, V, Bc)
got = backward(Q, K, V, O, dOm, m, l, Br, Bc)
want = naive_backward(Q, K, V, dOm)
print()
print(f"N = {N}, d = {d}, Br = {Br}, Bc = {Bc}")
for name, g, w in zip(("dQ", "dK", "dV"), got, want):
    print(f"max |{name} blocked - {name} naive| = {np.abs(g - w).max():.2e}")


# Finite-difference check of the naive gradient, so the reference is not circular.
def loss(Q, K, V):
    S = Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return float(((P @ V) * dOm).sum())


eps = 1e-6
fd = np.zeros_like(Q)
for a in range(N):
    for b in range(d):
        Qp, Qm = Q.copy(), Q.copy()
        Qp[a, b] += eps
        Qm[a, b] -= eps
        fd[a, b] = (loss(Qp, K, V) - loss(Qm, K, V)) / (2 * eps)
print(f"max |dQ blocked - dQ finite difference| = {np.abs(got[0] - fd).max():.2e}")
print(f"stored for the backward pass: O, l, m = {N * d + 2 * N} numbers; the attention matrix would be {N * N}")
