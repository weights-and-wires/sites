import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _causal(scores):
    T = scores.shape[-1]
    return np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -1e9)


def mla_naive(H, WQ, WDKV, WUK, WUV, n_h, d_h):
    """Expand keys and values from the latent, then attend (Equations 9-11, 7)."""
    raise NotImplementedError


def mla_absorbed(H, WQ, WDKV, WUK, WUV, n_h, d_h):
    """Attend over the latents C directly; never form K or V."""
    raise NotImplementedError
