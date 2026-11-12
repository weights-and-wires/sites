import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def depthwise_conv1d(x, w):
    """Depthwise convolution along time, (T, d) with (k, d) filters.

    Pad (k - 1) // 2 zeros at the start and k - 1 - (k - 1) // 2 at the end.
    """
    T, d = x.shape
    k = w.shape[0]
    left = (k - 1) // 2
    xp = np.pad(x, ((left, k - 1 - left), (0, 0)))
    out = np.zeros((T, d))
    for i in range(k):
        out += w[i] * xp[i:i + T]
    return out


def conv_module(x, w_pw1, w_dw, gamma, beta, w_pw2, eps=1e-5):
    """Conformer convolution module without dropout and the residual.

    Order: layer norm, pointwise conv (d -> 2d), GLU, depthwise conv,
    batch norm (batch statistics over time), Swish, pointwise conv.
    """
    d = x.shape[1]
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x = (x - mu) / np.sqrt(var + eps)
    h = x @ w_pw1
    g = h[:, :d] * sigmoid(h[:, d:])
    z = depthwise_conv1d(g, w_dw)
    z = (z - z.mean(axis=0)) / np.sqrt(z.var(axis=0) + eps) * gamma + beta
    z = z * sigmoid(z)
    return z @ w_pw2
