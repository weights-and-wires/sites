import numpy as np


def flatten_acoustic(Y, N):
    """Flatten Y of shape (T_A, Q), symbols in [0, N), to ids of length T_A * Q.

    Row-major order, with (i mod Q) * N added at flat position i.
    Raise ValueError if any symbol is outside [0, N).
    """
    raise NotImplementedError


def unflatten_acoustic(ids, Q, N):
    """Invert flatten_acoustic and return an array of shape (T_A, Q).

    Raise ValueError if len(ids) is not a multiple of Q.
    """
    raise NotImplementedError
