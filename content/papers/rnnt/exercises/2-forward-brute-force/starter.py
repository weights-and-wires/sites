import itertools

import numpy as np


def forward(emit, null):
    """Return (alpha, prob).

    emit: (T, U), emit[t, u] = y(t+1, u).
    null: (T, U+1), null[t, u] = null(t+1, u).
    alpha: (T, U+1) forward variables (Equation 16).
    prob: Pr(y | x) (Equation 17).
    """
    raise NotImplementedError


def brute_force(emit, null):
    """Return (prob, n): the sum over every alignment and how many there are."""
    raise NotImplementedError
