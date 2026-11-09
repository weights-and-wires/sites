import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Finished for you. Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v).

    mask: boolean, broadcastable to (..., n, m); True means "may attend".
    Returns (output, weights).
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=None):
    """Multi-head self-attention.

    X: (n, d_model); Wq, Wk, Wv, Wo: (d_model, d_model); h divides d_model.
    mask: optional boolean (n, n). Returns (n, d_model).
    """
    raise NotImplementedError
