import numpy as np


def flash_attention2_forward(Q, K, V, Br, Bc, tau=1.0):
    """Algorithm 1 of FlashAttention-2: outer loop over Q blocks.

    Q: (N, d), K: (M, d), V: (M, dv). Returns (O, L) with O of shape
    (N, dv) and the logsumexp L of shape (N,).
    """
    N, M = Q.shape[0], K.shape[0]
    O = np.zeros((N, V.shape[1]))
    L = np.zeros(N)
    for i in range(0, N, Br):
        Qi = Q[i:i + Br]
        O_t = np.zeros((Qi.shape[0], V.shape[1]))
        l = np.zeros(Qi.shape[0])
        m = np.full(Qi.shape[0], -np.inf)
        for j in range(0, M, Bc):
            S = tau * (Qi @ K[j:j + Bc].T)
            m_new = np.maximum(m, S.max(axis=1))
            P = np.exp(S - m_new[:, None])
            alpha = np.exp(m - m_new)
            l = alpha * l + P.sum(axis=1)
            O_t = alpha[:, None] * O_t + P @ V[j:j + Bc]
            m = m_new
        O[i:i + Br] = O_t / l[:, None]
        L[i:i + Br] = m + np.log(l)
    return O, L
