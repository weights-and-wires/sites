import numpy as np


def combine(left, right):
    """Compose two affine steps. `left` happens first.

    left = (a1, b1) is h -> a1 * h + b1, right = (a2, b2) is h -> a2 * h + b2.
    Returns the pair for "left, then right".
    """
    a1, b1 = left
    a2, b2 = right
    return a2 * a1, a2 * b1 + b2


def parallel_scan(a, b):
    """Return h with h[t] = a[t] * h[t-1] + b[t] and h[-1] = 0, for a and b of shape (L, ...).

    Use ceil(log2 L) rounds of vectorized combines (Hillis-Steele).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    L = a.shape[0]
    d = 1
    while d < L:
        new_a, new_b = combine((a[:-d], b[:-d]), (a[d:], b[d:]))
        a = np.concatenate([a[:d], new_a])
        b = np.concatenate([b[:d], new_b])
        d *= 2
    return b
