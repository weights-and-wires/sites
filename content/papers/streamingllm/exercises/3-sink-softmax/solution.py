import numpy as np


def softmax(x):
    """Stable softmax of a 1-D array."""
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / e.sum()


def softmax_one(x):
    """SoftMax_1 (Equation 2): exp(x_i) / (1 + sum_j exp(x_j)), without overflow or underflow."""
    x = np.asarray(x, dtype=float)
    m = max(np.max(x), 0.0)
    e = np.exp(x - m)
    return e / (np.exp(-m) + e.sum())


def sink_mass(scores, num_sinks):
    """Total softmax weight on the first num_sinks scores."""
    return float(softmax(scores)[:num_sinks].sum())


def attend(scores, V, evict=0):
    """Attention output after dropping the first `evict` tokens. Returns shape (d,)."""
    scores = np.asarray(scores, dtype=float)[evict:]
    V = np.asarray(V, dtype=float)[evict:]
    return softmax(scores) @ V
