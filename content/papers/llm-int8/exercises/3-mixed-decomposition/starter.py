import numpy as np


def outlier_dims(X, alpha=6.0):
    """Finished for you: column indices of X with at least one |value| >= alpha."""
    return np.where(np.max(np.abs(X), axis=0) >= alpha)[0]


def vectorwise_matmul(X, W):
    """Finished for you: int8 matmul with one constant per row of X and per column of W."""
    cx = 127.0 / np.max(np.abs(X), axis=1)
    cw = 127.0 / np.max(np.abs(W), axis=0)
    Xq = np.rint(X * cx[:, None]).astype(np.int8)
    Wq = np.rint(W * cw[None, :]).astype(np.int8)
    return (Xq.astype(np.int32) @ Wq.astype(np.int32)) / np.outer(cx, cw)


def llm_int8_matmul(X, W, alpha=6.0):
    """Equation 8: float matmul over the outlier dimensions plus int8 matmul over the rest."""
    raise NotImplementedError
