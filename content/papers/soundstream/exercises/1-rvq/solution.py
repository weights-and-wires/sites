import numpy as np


def rvq_encode(y, codebooks):
    """Residual vector quantization (Algorithm 1 in the paper).

    y: array of shape (S, D), the encoder output.
    codebooks: list of n_q arrays, each of shape (N, D).
    Returns (indices, y_hat). indices has shape (S, n_q) with integer entries,
    and y_hat has shape (S, D) and is the sum of the selected code vectors.
    """
    y = np.asarray(y, dtype=float)
    residual = y.copy()
    y_hat = np.zeros_like(y)
    cols = []
    for c in codebooks:
        d = ((residual[:, None, :] - c[None]) ** 2).sum(-1)
        idx = d.argmin(axis=1)
        q = c[idx]
        y_hat = y_hat + q
        residual = residual - q
        cols.append(idx)
    return np.stack(cols, axis=1), y_hat


def rvq_decode(indices, codebooks):
    """Rebuild y_hat from indices of shape (S, n) with n <= len(codebooks).

    Stage i uses codebooks[i]. Only the first n stages contribute.
    """
    indices = np.asarray(indices)
    out = np.zeros((indices.shape[0], codebooks[0].shape[1]))
    for i in range(indices.shape[1]):
        out = out + codebooks[i][indices[:, i]]
    return out
