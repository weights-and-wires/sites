import numpy as np


def causal_conv1d(x, w, dilation=1):
    """Causal dilated convolution of a 1-D signal.

    y[t] = sum_j w[j] * x[t - (k - 1 - j) * dilation], with x[i] = 0 for i < 0.
    Return an array with the same length as x.
    """
    raise NotImplementedError
