import numpy as np


def rope(x, positions, base=10000.0):
    """Rotate each coordinate pair of x by position * theta_j (Equation 34).

    x: (..., n, d) with d even. positions: (n,). Returns an array shaped like x.
    Pair j is (x[..., 2j], x[..., 2j + 1]) and rotates by positions[t] * base ** (-2j / d).
    """
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    angle = np.asarray(positions, dtype=float)[:, None] * theta[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out
