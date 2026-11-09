import numpy as np


def rope_angles(positions, d, base=10000.0):
    """Return the (n, d // 2) array of angles positions[t] * base ** (-2 j / d)."""
    raise NotImplementedError


def interpolate_positions(positions, train_len, new_len):
    """Rescale positions by train_len / new_len (position interpolation)."""
    raise NotImplementedError


def ntk_base(base, d, s):
    """Return the NTK-aware base: base * s ** (d / (d - 2))."""
    raise NotImplementedError
