import numpy as np


def chi_square_stat(counts, probs):
    """Finished for you: Pearson chi-square statistic of counts against probs."""
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(probs, dtype=float) * counts.sum()
    mask = expected > 0
    return float(np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask]))


def residual_distribution(p, q):
    """Return norm(max(0, p - q)) as a float array that sums to 1."""
    raise NotImplementedError


def speculative_sample(p, q, rng):
    """Return (token, accepted): one token distributed as p, built from a draft token drawn from q."""
    raise NotImplementedError
