import numpy as np


def softmax(x):
    """Stable softmax of a 1-D array."""
    raise NotImplementedError


def softmax_one(x):
    """SoftMax_1 (Equation 2): exp(x_i) / (1 + sum_j exp(x_j)), without overflow or underflow."""
    raise NotImplementedError


def sink_mass(scores, num_sinks):
    """Total softmax weight on the first num_sinks scores."""
    raise NotImplementedError


def attend(scores, V, evict=0):
    """Attention output after dropping the first `evict` tokens. Returns shape (d,)."""
    raise NotImplementedError
