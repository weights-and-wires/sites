import numpy as np


def boltzmann_policy(q, alpha=1.0):
    """Return pi(a) proportional to exp(q[a] / alpha), shape (A,), summing to 1."""
    raise NotImplementedError


def soft_value(q, alpha=1.0):
    """Return alpha * log(sum_a exp(q[a] / alpha)), computed without overflow."""
    raise NotImplementedError
