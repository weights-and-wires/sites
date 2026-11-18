import numpy as np


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm over the last axis, times weight."""
    raise NotImplementedError


def grouped_rmsnorm(x, weight, k, eps=1e-6):
    """RMSNorm computed separately over k contiguous groups of the last axis (Equation 3)."""
    raise NotImplementedError
