import numpy as np


def acceptance_rate(p, q):
    """alpha for one position: sum over x of min(p(x), q(x))."""
    return float(np.minimum(np.asarray(p, dtype=float), np.asarray(q, dtype=float)).sum())


def expected_tokens(alpha, gamma):
    """Expected tokens per run of Algorithm 1 (Equation 1). Handle alpha == 1."""
    if alpha == 1:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def walltime_speedup(alpha, gamma, c):
    """Expected walltime improvement factor (Theorem 3.8); c is the cost coefficient."""
    return expected_tokens(alpha, gamma) / (gamma * c + 1)


def best_gamma(alpha, c, max_gamma=40):
    """The integer gamma in [1, max_gamma] that maximizes walltime_speedup."""
    return max(range(1, max_gamma + 1), key=lambda g: walltime_speedup(alpha, g, c))
