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
    d = Q.shape[-1]
    mask = topk_mask(I, k)
    logits = np.einsum("thd,sd->hts", Q, K) / np.sqrt(d)
    logits = np.where(mask[None], logits, -np.inf)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = e / e.sum(axis=-1, keepdims=True)
    return np.einsum("hts,sv->thv", weights, V)
