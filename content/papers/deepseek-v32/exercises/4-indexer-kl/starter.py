import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def selected_keys(I_row, k, t):
    """Indices of the keys that row t keeps: all visible keys if k is None, else the top-k (given)."""
    if k is None or t + 1 <= k:
        return np.arange(t + 1)
    return np.argsort(-I_row)[:k]


def indexer_kl(attn, I, k=None):
    """Return the summed KL(p || softmax(I)) of Equation 3 (k is None) or Equation 4."""
    raise NotImplementedError
