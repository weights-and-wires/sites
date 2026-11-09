import numpy as np


def make_grid(W, bits):
    """Finished for you: per-row asymmetric min-max grid. Returns (scale, zero, maxq)."""
    maxq = 2**bits - 1
    lo = np.minimum(W.min(axis=1), 0.0)
    hi = np.maximum(W.max(axis=1), 0.0)
    scale = (hi - lo) / maxq
    zero = np.rint(-lo / scale)
    return scale, zero, maxq


def quant(w, scale, zero, maxq):
    """Finished for you: round w to the nearest grid point and dequantize."""
    q = np.clip(np.rint(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def inverse_factor(X, percdamp=0.01):
    """Finished for you: upper Cholesky factor U of (2 X X^T + lambda I)^-1, so H^-1 = U^T U.

    lambda is percdamp times the mean diagonal of 2 X X^T (Section 4, Step 3).
    """
    H = 2.0 * X @ X.T
    H = H + percdamp * np.mean(np.diag(H)) * np.eye(H.shape[0])
    return np.linalg.cholesky(np.linalg.inv(H)).T


def gptq_quantize(W, X, bits, percdamp=0.01):
    """Algorithm 1 with one block: quantize the columns of W left to right.

    W has shape (d_row, d_col) and X has shape (d_col, m). Return Q, the quantized
    and dequantized weights, with the same shape as W. Don't change W in place.
    """
    W = W.copy()
    d_row, d_col = W.shape
    scale, zero, maxq = make_grid(W, bits)
    U = inverse_factor(X, percdamp)
    Q = np.zeros_like(W)
    for j in range(d_col):
        Q[:, j] = quant(W[:, j], scale, zero, maxq)
        err = (W[:, j] - Q[:, j]) / U[j, j]
        W[:, j:] -= np.outer(err, U[j, j:])
    return Q
