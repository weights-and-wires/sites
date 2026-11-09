import numpy as np


def chi_square_stat(counts, probs):
    """Finished for you: Pearson chi-square statistic of counts against probs."""
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(probs, dtype=float) * counts.sum()
    mask = expected > 0
    return float(np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask]))


def residual_distribution(p, q):
    """Return norm(max(0, p - q)) as a float array that sums to 1."""
    r = np.maximum(0.0, np.asarray(p, dtype=float) - np.asarray(q, dtype=float))
    return r / r.sum()


def speculative_sample(p, q, rng):
    """Return (token, accepted): one token distributed as p, built from a draft token drawn from q."""
    x = int(rng.choice(len(q), p=q))
    if rng.random() <= min(1.0, p[x] / q[x]):
        return x, True
    return int(rng.choice(len(p), p=residual_distribution(p, q))), False
