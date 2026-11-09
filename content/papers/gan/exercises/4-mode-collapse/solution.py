import numpy as np


def histogram(samples, edges):
    """Return the probability mass per bin; samples outside the edges are dropped."""
    counts, _ = np.histogram(np.asarray(samples, dtype=float), bins=edges)
    return counts / counts.sum()


def _kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_divergence(p, q):
    """Return the Jensen-Shannon divergence (natural log) of two histograms."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = (p + q) / 2
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def modes_covered(samples, modes, radius, min_frac):
    """Return how many modes have at least min_frac of all samples within radius."""
    samples = np.asarray(samples, dtype=float)
    return int(sum(np.mean(np.abs(samples - c) < radius) >= min_frac for c in modes))
