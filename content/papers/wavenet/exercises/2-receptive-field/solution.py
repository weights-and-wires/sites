import numpy as np


def causal_conv1d(x, w, dilation=1):
    """Finished for you. y[t] = sum_j w[j] * x[t - (k - 1 - j) * dilation]."""
    k = len(w)
    y = np.zeros(len(x))
    for j in range(k):
        shift = (k - 1 - j) * dilation
        y[shift:] += w[j] * x[: len(x) - shift]
    return y


def stack_forward(x, dilations, kernel_size=2):
    """Finished for you. A linear stack of all-ones causal filters."""
    for d in dilations:
        x = causal_conv1d(x, np.ones(kernel_size), d)
    return x


def receptive_field(dilations, kernel_size=2):
    """Return the receptive field, in samples, from the closed-form formula."""
    return 1 + (kernel_size - 1) * sum(dilations)


def measured_receptive_field(dilations, kernel_size=2):
    """Return the receptive field by sending a unit impulse through stack_forward."""
    n = receptive_field(dilations, kernel_size) + 5
    x = np.zeros(n)
    x[0] = 1.0
    y = stack_forward(x, dilations, kernel_size)
    return int(np.nonzero(y)[0].max()) + 1
