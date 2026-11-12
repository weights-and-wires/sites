import numpy as np


def attention(x, Wq, Wk, Wv, causal=False):
    """Single-head scaled dot-product self-attention.

    Returns (out, weights). With causal=True, position i reads only j <= i.
    """
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.T / np.sqrt(q.shape[-1])
    if causal:
        keep = np.tril(np.ones(scores.shape)) == 1
        scores = np.where(keep, scores, -1e9)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    return w @ v, w
