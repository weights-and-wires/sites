import numpy as np


def softmax(x):
    """Return a numerically stable softmax of a 1-D array."""
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def additive_attention(s_prev, H, Wa, Ua, va):
    """Return (c, alpha) for one decoder step.

    s_prev: (n,) decoder state s_{i-1}
    H: (Tx, 2n) annotations, one per row
    Wa: (n_prime, n), Ua: (n_prime, 2n), va: (n_prime,)
    c: (2n,), alpha: (Tx,)
    """
    raise NotImplementedError
