import numpy as np


def softmax(logits):
    """Return a probability vector of the same shape as logits."""
    x = np.asarray(logits, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def sl_grad(logits, a):
    """Return d log softmax(logits)[a] / d logits, an array of shape (K,)."""
    p = softmax(logits)
    e = np.zeros_like(p)
    e[a] = 1.0
    return e - p


def rl_grad(logits, a, z, baseline=0.0):
    """Return (z - baseline) * d log softmax(logits)[a] / d logits."""
    return (z - baseline) * sl_grad(logits, a)
