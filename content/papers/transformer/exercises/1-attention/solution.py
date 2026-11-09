import numpy as np


def softmax(x, axis=-1):
    """Return a numerically stable softmax of x along axis."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Return (output, weights).

    Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v)
    mask: boolean array broadcastable to (..., n, m); True means "may attend".
    output: (..., n, d_v), weights: (..., n, m)
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights
