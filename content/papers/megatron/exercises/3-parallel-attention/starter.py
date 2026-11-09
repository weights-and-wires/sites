import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def local_attention(X, Wq_r, Wk_r, Wv_r, local_heads, d_k, mask=None):
    """Finished for you: attention for the heads in one GPU's column slices.

    Returns an array of shape (tokens, local_heads * d_k).
    """
    n = X.shape[0]
    split = lambda M: M.reshape(n, local_heads, d_k).transpose(1, 0, 2)
    Q, K, V = split(X @ Wq_r), split(X @ Wk_r), split(X @ Wv_r)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    out = softmax(scores) @ V
    return out.transpose(1, 0, 2).reshape(n, local_heads * d_k)


def tp_attention(X, Wq, Wk, Wv, Wo, heads, n, mask=None):
    """Tensor-parallel multi-head self-attention on n simulated GPUs."""
    raise NotImplementedError
