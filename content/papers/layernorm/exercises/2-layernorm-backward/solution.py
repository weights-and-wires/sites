import numpy as np


def layer_norm_forward(x, gain, bias, eps=1e-5):
    """Finished for you. x has shape (N, H). Returns (y, cache)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    xhat = (x - mu) / sigma
    return gain * xhat + bias, (xhat, sigma, gain)


def layer_norm_backward(dy, cache):
    """Return (dx, dgain, dbias) for dy of shape (N, H).

    dx has shape (N, H); dgain and dbias have shape (H,).
    """
    xhat, sigma, gain = cache
    dxhat = dy * gain
    dx = (dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)) / sigma
    dgain = (dy * xhat).sum(axis=0)
    dbias = dy.sum(axis=0)
    return dx, dgain, dbias
