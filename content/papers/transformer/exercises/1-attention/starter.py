import numpy as np


def softmax(x, axis=-1):
    """Return a numerically stable softmax of x along axis."""
    raise NotImplementedError


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Return (output, weights).

    Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v)
    mask: boolean array broadcastable to (..., n, m); True means "may attend".
    output: (..., n, d_v), weights: (..., n, m)
    """
    raise NotImplementedError
