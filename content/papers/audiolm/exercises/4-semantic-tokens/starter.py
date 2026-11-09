import numpy as np


def semantic_tokens(feats, centroids, mean, std):
    """Normalize feats (T, D) with mean and std, then return the index of the
    nearest centroid (K, D) in squared Euclidean distance, as an int array (T,)."""
    raise NotImplementedError


def collapse_repeats(tokens):
    """Remove consecutive repeats. Return (kept, run_lengths) as int arrays."""
    raise NotImplementedError
