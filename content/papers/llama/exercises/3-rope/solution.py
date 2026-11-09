import numpy as np


def apply_rope(x, positions, base=10000.0):
    """Rotate adjacent coordinate pairs of each row by position-dependent angles.

    x: (n, d) with d even; positions: (n,). Pair j of row i turns by
    positions[i] * base ** (-2 j / d). Returns an array with the shape of x.
    """
    n, d = x.shape
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    angles = np.asarray(positions, dtype=float)[:, None] * theta[None, :]
    c, s = np.cos(angles), np.sin(angles)
    out = np.empty_like(x, dtype=float)
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out
