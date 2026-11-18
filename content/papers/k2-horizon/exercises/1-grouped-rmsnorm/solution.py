import numpy as np


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm over the last axis, times weight."""
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * weight


def grouped_rmsnorm(x, weight, k, eps=1e-6):
    """RMSNorm computed separately over k contiguous groups of the last axis (Equation 3)."""
    d = x.shape[-1]
    if d % k:
        raise ValueError(f"d={d} is not divisible by k={k}")
    g = x.reshape(*x.shape[:-1], k, d // k)
    g = g / np.sqrt(np.mean(g**2, axis=-1, keepdims=True) + eps)
    return g.reshape(x.shape) * weight
