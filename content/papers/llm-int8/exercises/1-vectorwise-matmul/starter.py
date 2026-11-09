import numpy as np


def quantize_rows(A):
    """Return (Aq, c): int8 codes and the per-row constants c = 127 / max|row|."""
    raise NotImplementedError


def vectorwise_matmul(X, W):
    """Approximate X @ W with int8 codes: one constant per row of X, one per column of W."""
    raise NotImplementedError
