import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _causal(scores):
    T = scores.shape[-1]
    return np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -1e9)


def mla_naive(H, WQ, WDKV, WUK, WUV, n_h, d_h):
    """Expand keys and values from the latent, then attend (Equations 9-11, 7)."""
    T = H.shape[0]
    C = H @ WDKV.T
    Q = (H @ WQ.T).reshape(T, n_h, d_h)
    K = (C @ WUK.T).reshape(T, n_h, d_h)
    V = (C @ WUV.T).reshape(T, n_h, d_h)
    out = np.zeros((T, n_h, d_h))
    for i in range(n_h):
        scores = Q[:, i] @ K[:, i].T / np.sqrt(d_h)
        out[:, i] = softmax(_causal(scores)) @ V[:, i]
    return out.reshape(T, n_h * d_h)


def mla_absorbed(H, WQ, WDKV, WUK, WUV, n_h, d_h):
    """Attend over the latents C directly; never form K or V."""
    T = H.shape[0]
    C = H @ WDKV.T
    Q = (H @ WQ.T).reshape(T, n_h, d_h)
    out = np.zeros((T, n_h, d_h))
    for i in range(n_h):
        WUK_i = WUK[i * d_h:(i + 1) * d_h]
        WUV_i = WUV[i * d_h:(i + 1) * d_h]
        q_abs = Q[:, i] @ WUK_i
        scores = q_abs @ C.T / np.sqrt(d_h)
        ctx = softmax(_causal(scores)) @ C
        out[:, i] = ctx @ WUV_i.T
    return out.reshape(T, n_h * d_h)
