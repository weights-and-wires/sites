import numpy as np


def layer_norm(x, gain, bias, eps=1e-5):
    """Layer normalization over the last axis.

    x:    array of shape (..., H)
    gain: array of shape (H,)
    bias: array of shape (H,)
    Returns an array with the shape of x.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gain * (x - mu) / np.sqrt(var + eps) + bias
