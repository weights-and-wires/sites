import numpy as np


def combine(left, right):
    """Compose two affine steps. `left` happens first.

    left = (a1, b1) is h -> a1 * h + b1, right = (a2, b2) is h -> a2 * h + b2.
    Returns the pair for "left, then right".
    """
    raise NotImplementedError


def parallel_scan(a, b):
    """Return h with h[t] = a[t] * h[t-1] + b[t] and h[-1] = 0, for a and b of shape (L, ...).

    Use ceil(log2 L) rounds of vectorized combines (Hillis-Steele).
    """
    raise NotImplementedError
