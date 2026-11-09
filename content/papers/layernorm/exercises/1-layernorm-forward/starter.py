import numpy as np


def layer_norm(x, gain, bias, eps=1e-5):
    """Layer normalization over the last axis.

    x:    array of shape (..., H)
    gain: array of shape (H,)
    bias: array of shape (H,)
    Returns an array with the shape of x.
    """
    raise NotImplementedError
