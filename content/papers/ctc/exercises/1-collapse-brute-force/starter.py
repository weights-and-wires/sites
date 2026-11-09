import itertools

import numpy as np


def collapse(path, blank=0):
    """Return B(path) as a tuple: merge repeated symbols, then remove blanks."""
    raise NotImplementedError


def brute_force_ctc(y, label, blank=0):
    """Return p(label | x) by summing the path probabilities over all K**T paths.

    y: array of shape (T, K); each row is a distribution over the K symbols.
    label: tuple of integers that are not the blank.
    """
    raise NotImplementedError
