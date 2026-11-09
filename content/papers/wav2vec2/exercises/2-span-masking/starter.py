import numpy as np


def sample_starts(T, p, rng):
    """Return a sorted array of round(p * T) distinct start indices from range(T)."""
    raise NotImplementedError


def span_mask(T, starts, M):
    """Return a boolean array of length T, True where some span [s, s + M) covers the index."""
    raise NotImplementedError
