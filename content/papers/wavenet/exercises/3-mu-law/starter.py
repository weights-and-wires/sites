import numpy as np


def mu_law_encode(x, mu=255):
    """Map floats in [-1, 1] to integers in 0..mu."""
    raise NotImplementedError


def mu_law_decode(q, mu=255):
    """Map integers in 0..mu back to floats in [-1, 1]."""
    raise NotImplementedError
