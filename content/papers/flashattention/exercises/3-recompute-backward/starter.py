import numpy as np


def block_backward(Qi, Kj, Vj, Oi, dOi, mi, li):
    """Lines 11-22 of Algorithm 4 for one (i, j) block, with tau = 1.

    Returns (dQ_part, dK_part, dV_part) with shapes (Br, d), (Bc, d), (Bc, d).
    """
    raise NotImplementedError


def flash_attention_backward(Q, K, V, O, dO, m, l, Br, Bc):
    """Finished for you: Algorithm 4's loops. Calls block_backward on each block."""
    N, d = Q.shape
    M = K.shape[0]
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for j in range(0, M, Bc):
        for i in range(0, N, Br):
            dq, dk, dv = block_backward(
                Q[i:i + Br], K[j:j + Bc], V[j:j + Bc], O[i:i + Br], dO[i:i + Br], m[i:i + Br], l[i:i + Br]
            )
            dQ[i:i + Br] += dq
            dK[j:j + Bc] += dk
            dV[j:j + Bc] += dv
    return dQ, dK, dV
