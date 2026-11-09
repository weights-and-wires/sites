import numpy as np


def softmax(x, axis=-1):
    """Finished for you: a numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def cross_attention(phi, tau, W_Q, W_K, W_V):
    """Return (output, weights).

    phi: (N, d_eps) U-Net features; tau: (M, d_tau) condition tokens.
    W_Q: (d, d_eps); W_K, W_V: (d, d_tau).
    output: (N, d); weights: (N, M).
    """
    Q = phi @ W_Q.T
    K = tau @ W_K.T
    V = tau @ W_V.T
    d = Q.shape[-1]
    weights = softmax(Q @ K.T / np.sqrt(d), axis=-1)
    return weights @ V, weights
