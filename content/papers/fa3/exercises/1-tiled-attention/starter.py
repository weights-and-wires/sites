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
    raise NotImplementedError
