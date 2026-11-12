import numpy as np


def naive_attention(Q, K, V, tau=1.0):
    """Reference: materialize the scores and softmax over each row."""
    S = tau * Q @ K.T
    P = np.exp(S - S.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return P @ V


def tiled_attention(Q, K, V, Bc, tau=1.0):
    """Return (O, L) for one query block, streaming K and V in blocks of Bc rows.

    O = softmax(tau * Q K^T) V and L = m + log(l), the row logsumexp.
    """
    Br = Q.shape[0]
    O = np.zeros((Br, V.shape[1]))
    l = np.zeros(Br)
    m = np.full(Br, -np.inf)
    for j in range(0, K.shape[0], Bc):
        S = tau * Q @ K[j:j + Bc].T
        m_new = np.maximum(m, S.max(axis=1))
        P = np.exp(S - m_new[:, None])
        alpha = np.exp(m - m_new)
        l = alpha * l + P.sum(axis=1)
        O = alpha[:, None] * O + P @ V[j:j + Bc]
        m = m_new
    return O / l[:, None], m + np.log(l)
