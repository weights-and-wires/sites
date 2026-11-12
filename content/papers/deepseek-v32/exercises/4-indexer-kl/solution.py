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
    p = attn.sum(axis=0)
    p = p / p.sum(axis=-1, keepdims=True)
    total = 0.0
    for t in range(I.shape[0]):
        S = selected_keys(I[t], k, t)
        pS = p[t, S] / p[t, S].sum()
        qS = softmax(I[t, S])
        pos = pS > 0
        total += float(np.sum(pS[pos] * (np.log(pS[pos]) - np.log(qS[pos]))))
    return total
