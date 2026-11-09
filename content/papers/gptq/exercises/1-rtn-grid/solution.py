import numpy as np


def make_grid(W, bits):
    """Per-row asymmetric min-max grid. Return (scale, zero, maxq); scale and zero have shape (d_row,)."""
    maxq = 2**bits - 1
    lo = np.minimum(W.min(axis=1), 0.0)
    hi = np.maximum(W.max(axis=1), 0.0)
    scale = (hi - lo) / maxq
    zero = np.rint(-lo / scale)
    return scale, zero, maxq


def quant(w, scale, zero, maxq):
    """Round w to the nearest grid point and return the dequantized value (same shape as w)."""
    q = np.clip(np.rint(w / scale) + zero, 0, maxq)
    return scale * (q - zero)


def rtn(W, bits):
    """Round every weight of W to its row's grid."""
    scale, zero, maxq = make_grid(W, bits)
    return quant(W, scale[:, None], zero[:, None], maxq)
