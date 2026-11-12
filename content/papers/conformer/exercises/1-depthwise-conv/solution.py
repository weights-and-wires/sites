import numpy as np


def depthwise_conv1d(x, w):
    """Depthwise 1-D convolution along time with zero padding.

    x: (T, d) array. w: (k, d) array, one filter per channel.
    Pad (k - 1) // 2 zeros at the start and k - 1 - (k - 1) // 2 at the end.
    Return a (T, d) array.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    T, d = x.shape
    k = w.shape[0]
    left = (k - 1) // 2
    right = k - 1 - left
    xp = np.pad(x, ((left, right), (0, 0)))
    out = np.zeros((T, d))
    for i in range(k):
        out += w[i] * xp[i:i + T]
    return out
