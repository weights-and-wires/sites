import numpy as np


def silu(z):
    """z * sigmoid(z)."""
    return z / (1 + np.exp(-z))


def route(x, W_r, k):
    """Top-k experts per token (best first) and the softmax of their logits."""
    logits = x @ W_r
    idx = np.argsort(-logits, axis=1)[:, :k]
    top = np.take_along_axis(logits, idx, axis=1)
    p = np.exp(top - top.max(axis=1, keepdims=True))
    return idx, p / p.sum(axis=1, keepdims=True)


def mova_values(x, W_r, W_v, k):
    """Equation 6: sum over chosen experts of p_i * SiLU(x W_v[i]). Shape (T, d_v)."""
    idx, p = route(x, W_r, k)
    outs = silu(np.einsum("td,tkdv->tkv", x, W_v[idx]))
    return np.einsum("tk,tkv->tv", p, outs)
