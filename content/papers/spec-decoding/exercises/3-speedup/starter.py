import numpy as np


def acceptance_rate(p, q):
    """alpha for one position: sum over x of min(p(x), q(x))."""
    raise NotImplementedError


def expected_tokens(alpha, gamma):
    """Expected tokens per run of Algorithm 1 (Equation 1). Handle alpha == 1."""
    raise NotImplementedError


def walltime_speedup(alpha, gamma, c):
    """Expected walltime improvement factor (Theorem 3.8); c is the cost coefficient."""
    raise NotImplementedError


def best_gamma(alpha, c, max_gamma=40):
    """The integer gamma in [1, max_gamma] that maximizes walltime_speedup."""
    raise NotImplementedError
