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


def gptq_blocked(W, X, bits, blocksize=128, percdamp=0.01):
    """Algorithm 1: quantize W in blocks of `blocksize` columns with lazy batch updates.

    Inside a block, update only the columns of that block. After the block, push the
    accumulated error into all later columns with one matrix product. Return Q with the
    same shape as W. Don't change W in place.
    """
    W = W.copy()
    d_row, d_col = W.shape
    scale, zero, maxq = make_grid(W, bits)
    U = inverse_factor(X, percdamp)
    Q = np.zeros_like(W)
    for i in range(0, d_col, blocksize):
        i2 = min(i + blocksize, d_col)
        E = np.zeros((d_row, i2 - i))
        for j in range(i, i2):
            Q[:, j] = quant(W[:, j], scale, zero, maxq)
            E[:, j - i] = (W[:, j] - Q[:, j]) / U[j, j]
            W[:, j:i2] -= np.outer(E[:, j - i], U[j, j:i2])
        W[:, i2:] -= E @ U[i:i2, i2:]
    return Q
