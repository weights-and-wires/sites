import numpy as np


def topk_mask(I, k):
    """Boolean (L, L) mask of the top-k visible keys per row (given)."""
    L = I.shape[0]
    mask = np.zeros((L, L), dtype=bool)
    for t in range(L):
        n = min(k, t + 1)
        mask[t, np.argsort(-I[t])[:n]] = True
    return mask


def dsa_attention(Q, K, V, I, k):
    """Return the (L, H, d_v) attention output over the top-k keys of each query."""
    raise NotImplementedError
