import numpy as np


def topk_mask(I, k):
    """Return a boolean (L, L) mask that keeps the top-k visible keys of each row."""
    L = I.shape[0]
    mask = np.zeros((L, L), dtype=bool)
    for t in range(L):
        n = min(k, t + 1)
        mask[t, np.argsort(-I[t])[:n]] = True
    return mask
