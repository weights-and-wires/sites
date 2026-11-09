import numpy as np


def flatten_acoustic(Y, N):
    """Flatten Y of shape (T_A, Q), symbols in [0, N), to ids of length T_A * Q.

    Row-major order, with (i mod Q) * N added at flat position i.
    Raise ValueError if any symbol is outside [0, N).
    """
    Y = np.asarray(Y)
    if Y.min() < 0 or Y.max() >= N:
        raise ValueError(f"symbols must be in [0, {N})")
    Q = Y.shape[1]
    flat = Y.reshape(-1)
    return flat + (np.arange(flat.size) % Q) * N


def unflatten_acoustic(ids, Q, N):
    """Invert flatten_acoustic and return an array of shape (T_A, Q).

    Raise ValueError if len(ids) is not a multiple of Q.
    """
    ids = np.asarray(ids)
    if ids.size % Q != 0:
        raise ValueError("length must be a multiple of Q")
    return ids.reshape(-1, Q) - np.arange(Q) * N
