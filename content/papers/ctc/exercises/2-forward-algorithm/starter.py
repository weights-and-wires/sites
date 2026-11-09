import numpy as np


def ctc_forward(y, label, blank=0):
    """Return (alpha, p) for the labelling `label` and the output matrix y.

    y: array of shape (T, K); y[t, k] is the probability of symbol k at frame t.
    label: tuple of integers that are not the blank.
    alpha: array of shape (T, 2 * len(label) + 1), unscaled (Equations 5 to 7).
    p: p(label | x), from Equation 8.
    """
    raise NotImplementedError
