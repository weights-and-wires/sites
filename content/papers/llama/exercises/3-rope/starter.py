import numpy as np


def apply_rope(x, positions, base=10000.0):
    """Rotate adjacent coordinate pairs of each row by position-dependent angles.

    x: (n, d) with d even; positions: (n,). Pair j of row i turns by
    positions[i] * base ** (-2 j / d). Returns an array with the shape of x.
    """
    raise NotImplementedError
