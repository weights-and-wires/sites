import itertools

import numpy as np


def collapse(path, blank=0):
    """Return B(path) as a tuple: merge repeated symbols, then remove blanks."""
    out = []
    prev = None
    for s in path:
        if s != prev and s != blank:
            out.append(int(s))
        prev = s
    return tuple(out)


def brute_force_ctc(y, label, blank=0):
    """Return p(label | x) by summing the path probabilities over all K**T paths.

    y: array of shape (T, K); each row is a distribution over the K symbols.
    label: tuple of integers that are not the blank.
    """
    T, K = y.shape
    label = tuple(label)
    total = 0.0
    for path in itertools.product(range(K), repeat=T):
        if collapse(path, blank) == label:
            total += float(np.prod([y[t, k] for t, k in enumerate(path)]))
    return total
