import numpy as np


def boltzmann_policy(q, alpha=1.0):
    """Return pi(a) proportional to exp(q[a] / alpha), shape (A,), summing to 1."""
    z = np.asarray(q, dtype=float) / alpha
    e = np.exp(z - np.max(z))
    return e / e.sum()


def soft_value(q, alpha=1.0):
    """Return alpha * log(sum_a exp(q[a] / alpha)), computed without overflow."""
    z = np.asarray(q, dtype=float) / alpha
    m = np.max(z)
    return float(alpha * (m + np.log(np.sum(np.exp(z - m)))))
