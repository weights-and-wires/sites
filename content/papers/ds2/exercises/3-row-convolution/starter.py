import numpy as np


def row_convolution(h, W):
    """Row convolution (Equation 11). `h` has shape (T, d) and `W` has shape (d, tau + 1).

    Return r with shape (T, d), where r[t, i] = sum_j W[i, j] * h[t + j, i] (0-based j),
    and h[t + j, i] is 0 when t + j >= T.
    """
    raise NotImplementedError
