import numpy as np


def best_path_decode(y, blank=0):
    """Return B(pi*) as a tuple, where pi* is the most active symbol at each frame.

    y: array of shape (T, K).
    """
    raise NotImplementedError


def path_probability(y, path):
    """Return p(path | x) = the product over t of y[t, path[t]] (Equation 2)."""
    raise NotImplementedError
