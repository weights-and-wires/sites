import numpy as np


def bootstrap_ci(x, idx, alpha=0.05):
    """Percentile bootstrap interval for the mean of x.

    x: shape (n_seeds,). idx: integer array of shape (B, n), one resample per row.
    Returns (lo, hi), the alpha/2 and 1 - alpha/2 percentiles of the B resample means.
    """
    means = np.asarray(x, dtype=float)[np.asarray(idx)].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
