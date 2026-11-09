import numpy as np


def softmax(x, axis=-1):
    """Return a numerically stable softmax of x along axis."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def alignment_matrix(S_prev, H, Wa, Ua, va):
    """Return (alpha, C) for every decoder step.

    S_prev: (Ty, n), row i is s_{i-1}
    H: (Tx, 2n)
    Wa: (n_prime, n), Ua: (n_prime, 2n), va: (n_prime,)
    alpha: (Ty, Tx), C: (Ty, 2n)
    """
    raise NotImplementedError
