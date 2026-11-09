import numpy as np


def rope(x, pos, base=10000.0):
    """Rotate pair (x[2j], x[2j+1]) by pos * base ** (-2j / d). x: (..., T, d)."""
    raise NotImplementedError


def decoupled_scores(qC, kC, qR, kR, d_h, d_hR):
    """Pre-softmax scores, shape (n_h, T, T), from Equations 14-18."""
    raise NotImplementedError
