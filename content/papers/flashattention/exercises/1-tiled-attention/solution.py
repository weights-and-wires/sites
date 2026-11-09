import numpy as np


def flash_attention_forward(Q, K, V, Br, Bc, tau=1.0):
    """Algorithm 1: tiled attention with an online softmax.

    Q: (N, d), K: (M, d), V: (M, dv). Returns (O, m, l) with O of shape
    (N, dv), and m and l of shape (N,).
    """
    N = Q.shape[0]
    M = K.shape[0]
    O = np.zeros((N, V.shape[1]))
    l = np.zeros(N)
    m = np.full(N, -np.inf)
    for j in range(0, M, Bc):
        Kj, Vj = K[j:j + Bc], V[j:j + Bc]
        for i in range(0, N, Br):
            S = tau * (Q[i:i + Br] @ Kj.T)
            m_t = S.max(axis=1)
            P_t = np.exp(S - m_t[:, None])
            l_t = P_t.sum(axis=1)
            m_i, l_i, O_i = m[i:i + Br], l[i:i + Br], O[i:i + Br]
            m_new = np.maximum(m_i, m_t)
            old = np.exp(m_i - m_new)
            new = np.exp(m_t - m_new)
            l_new = old * l_i + new * l_t
            O[i:i + Br] = ((old * l_i)[:, None] * O_i + new[:, None] * (P_t @ Vj)) / l_new[:, None]
            l[i:i + Br], m[i:i + Br] = l_new, m_new
    return O, m, l
