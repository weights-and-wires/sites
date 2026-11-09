import numpy as np


def causal_conv(x, W, dilation=1):
    """Finished for you.

    x: (C_in, T). W: (k, C_out, C_in); W[j] applies to x[:, t - (k-1-j)*dilation].
    Returns (C_out, T), with zeros before t = 0.
    """
    k, c_out, _ = W.shape
    T = x.shape[1]
    y = np.zeros((c_out, T))
    for j in range(k):
        shift = (k - 1 - j) * dilation
        shifted = np.zeros_like(x)
        shifted[:, shift:] = x[:, : T - shift]
        y += W[j] @ shifted
    return y


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


def gated_activation(x, Wf, Wg, dilation=1):
    """Return tanh(Wf * x) * sigmoid(Wg * x), with * a causal dilated convolution."""
    return np.tanh(causal_conv(x, Wf, dilation)) * sigmoid(causal_conv(x, Wg, dilation))
