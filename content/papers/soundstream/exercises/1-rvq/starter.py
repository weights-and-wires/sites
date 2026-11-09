import numpy as np


def rvq_encode(y, codebooks):
    """Residual vector quantization (Algorithm 1 in the paper).

    y: array of shape (S, D), the encoder output.
    codebooks: list of n_q arrays, each of shape (N, D).
    Returns (indices, y_hat). indices has shape (S, n_q) with integer entries,
    and y_hat has shape (S, D) and is the sum of the selected code vectors.
    """
    raise NotImplementedError


def rvq_decode(indices, codebooks):
    """Rebuild y_hat from indices of shape (S, n) with n <= len(codebooks).

    Stage i uses codebooks[i]. Only the first n stages contribute.
    """
    raise NotImplementedError
