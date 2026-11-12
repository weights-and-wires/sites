import numpy as np


def depthwise_conv1d(x, w):
    """Depthwise 1-D convolution along time with zero padding.

    x: (T, d) array. w: (k, d) array, one filter per channel.
    Pad (k - 1) // 2 zeros at the start and k - 1 - (k - 1) // 2 at the end.
    Return a (T, d) array.
    """
    raise NotImplementedError
