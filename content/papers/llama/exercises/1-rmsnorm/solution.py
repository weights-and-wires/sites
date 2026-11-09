import numpy as np


def rmsnorm(x, g, eps=1e-6):
    """RMSNorm over the last axis.

    x: (..., d); g: (d,). Returns an array with the shape of x:
    x / sqrt(mean(x^2) + eps) * g, with the mean taken over the last axis.
    """
    ms = np.mean(x**2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * g
