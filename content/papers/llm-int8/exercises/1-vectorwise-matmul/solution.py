import numpy as np


def quantize_rows(A):
    """Return (Aq, c): int8 codes and the per-row constants c = 127 / max|row|."""
    c = 127.0 / np.max(np.abs(A), axis=1)
    Aq = np.rint(A * c[:, None]).astype(np.int8)
    return Aq, c


def vectorwise_matmul(X, W):
    """Approximate X @ W with int8 codes: one constant per row of X, one per column of W."""
    Xq, cx = quantize_rows(X)
    Wt, cw = quantize_rows(W.T)
    Wq = Wt.T
    C32 = Xq.astype(np.int32) @ Wq.astype(np.int32)
    return C32 / np.outer(cx, cw)
