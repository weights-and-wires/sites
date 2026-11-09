import numpy as np


def best_path_decode(y, blank=0):
    """Return B(pi*) as a tuple, where pi* is the most active symbol at each frame.

    y: array of shape (T, K).
    """
    path = np.argmax(y, axis=1)
    out = []
    prev = None
    for s in path:
        if s != prev and s != blank:
            out.append(int(s))
        prev = s
    return tuple(out)


def path_probability(y, path):
    """Return p(path | x) = the product over t of y[t, path[t]] (Equation 2)."""
    return float(np.prod([y[t, k] for t, k in enumerate(path)]))
