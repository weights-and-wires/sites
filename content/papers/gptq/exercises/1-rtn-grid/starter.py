import numpy as np


def make_grid(W, bits):
    """Per-row asymmetric min-max grid. Return (scale, zero, maxq); scale and zero have shape (d_row,)."""
    raise NotImplementedError


def quant(w, scale, zero, maxq):
    """Round w to the nearest grid point and return the dequantized value (same shape as w)."""
    raise NotImplementedError


def rtn(W, bits):
    """Round every weight of W to its row's grid."""
    scale, zero, maxq = make_grid(W, bits)
    return quant(W, scale[:, None], zero[:, None], maxq)
