"""Shared NumPy helpers for the GPTQ page scripts.

Notation follows the paper (Section 3 and Algorithm 1): W has shape
(d_row, d_col), X has shape (d_col, m), H = 2 X X^T, and the quantized
weights are Q.
"""

import numpy as np


def make_grid(W, bits):
    """Per-row asymmetric min-max grid (Section 5, Setup). Returns scale, zero, maxq."""
    maxq = 2**bits - 1
    lo = np.minimum(W.min(axis=1), 0.0)
    hi = np.maximum(W.max(axis=1), 0.0)
    scale = (hi - lo) / maxq
    zero = np.rint(-lo / scale)
    return scale, zero, maxq


def quant(w, scale, zero, maxq):
    """Round w to the nearest grid point; w, scale, and zero are broadcast together."""
    q = np.clip(np.rint(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def rtn(W, bits):
    scale, zero, maxq = make_grid(W, bits)
    return quant(W, scale[:, None], zero[:, None], maxq)


def inverse_factor(X, percdamp=0.01):
    """Upper Cholesky factor U of (2 X X^T + lambda I)^-1, with H^-1 = U^T U."""
    H = 2.0 * X @ X.T
    H = H + percdamp * np.mean(np.diag(H)) * np.eye(H.shape[0])
    return np.linalg.cholesky(np.linalg.inv(H)).T


def gptq(W, X, bits, blocksize=None, percdamp=0.01):
    """Algorithm 1 of the paper. blocksize=None means one block of all columns."""
    W = W.copy()
    d_row, d_col = W.shape
    B = d_col if blocksize is None else blocksize
    scale, zero, maxq = make_grid(W, bits)
    U = inverse_factor(X, percdamp)
    Q = np.zeros_like(W)
    for i in range(0, d_col, B):
        i2 = min(i + B, d_col)
        E = np.zeros((d_row, i2 - i))
        for j in range(i, i2):
            Q[:, j] = quant(W[:, j], scale, zero, maxq)
            E[:, j - i] = (W[:, j] - Q[:, j]) / U[j, j]
            W[:, j:i2] -= np.outer(E[:, j - i], U[j, j:i2])
        W[:, i2:] -= E @ U[i:i2, i2:]
    return Q


def layer_loss(W, Q, X):
    """Equation 1: ||W X - Q X||_F^2."""
    return float(np.sum(((W - Q) @ X) ** 2))


def correlated_inputs(rng, d_col, m, rank=4, noise=0.3):
    """Synthetic layer inputs: a few shared directions plus noise, so H is far from diagonal."""
    return rng.normal(size=(d_col, rank)) @ rng.normal(size=(rank, m)) + noise * rng.normal(size=(d_col, m))
