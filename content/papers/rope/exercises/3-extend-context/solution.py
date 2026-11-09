import numpy as np


def rope_angles(positions, d, base=10000.0):
    """Return the (n, d // 2) array of angles positions[t] * base ** (-2 j / d)."""
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    return np.asarray(positions, dtype=float)[:, None] * theta[None, :]


def interpolate_positions(positions, train_len, new_len):
    """Rescale positions by train_len / new_len (position interpolation)."""
    return np.asarray(positions, dtype=float) * (train_len / new_len)


def ntk_base(base, d, s):
    """Return the NTK-aware base: base * s ** (d / (d - 2))."""
    return base * s ** (d / (d - 2))
