import numpy as np


def difficulty_bins(pass1, n_bins=5):
    """Return bin indices, 0 for the easiest (highest pass@1) up to n_bins - 1."""
    raise NotImplementedError


def compute_optimal(bins, outcomes, n_bins=5):
    """Return (best strategy name per bin, overall accuracy of per-bin choices)."""
    raise NotImplementedError
