import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Finished for you. Q: (..., n, d_k), K: (..., m, d_k), V: (..., m, d_v).

    mask: boolean, broadcastable to (..., n, m); True means "may attend".
    Returns (output, weights).
    """
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def gqa_attention(X, Wq, Wk, Wv, Wo, H, G, mask=None):
    """Grouped-query self-attention with H query heads and G key-value heads.

    X: (n, d_model); Wq: (d_model, H * d_k); Wk, Wv: (d_model, G * d_k);
    Wo: (H * d_k, d_model); G divides H. Query head i reads group i // (H // G).
    mask: optional boolean (n, n). Returns (n, d_model).
    """
    n = X.shape[0]
    d_k = Wq.shape[1] // H
    r = H // G

    def split(M, heads):
        # (n, heads * d_k) -> (heads, n, d_k)
        return M.reshape(n, heads, d_k).transpose(1, 0, 2)

    Q = split(X @ Wq, H)
    K = np.repeat(split(X @ Wk, G), r, axis=0)  # head i gets group i // r
    V = np.repeat(split(X @ Wv, G), r, axis=0)
    heads, _ = scaled_dot_product_attention(Q, K, V, mask)
    concat = heads.transpose(1, 0, 2).reshape(n, H * d_k)
    return concat @ Wo
