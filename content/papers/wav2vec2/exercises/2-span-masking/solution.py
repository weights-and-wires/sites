import numpy as np


def sample_starts(T, p, rng):
    """Return a sorted array of round(p * T) distinct start indices from range(T)."""
    n = int(round(p * T))
    return np.sort(rng.choice(T, size=n, replace=False))


def span_mask(T, starts, M):
    """Return a boolean array of length T, True where some span [s, s + M) covers the index."""
    mask = np.zeros(T, dtype=bool)
    for s in starts:
        mask[s:s + M] = True
    return mask
