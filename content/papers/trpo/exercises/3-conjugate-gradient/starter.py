import numpy as np


def conjugate_gradient(fvp, b, iters=10, tol=1e-10):
    """Approximately solve A x = b using only the function fvp(v) = A v.

    Start from x = 0. Stop after `iters` iterations, or earlier when the squared
    residual norm r.r drops below `tol`. Return x.
    """
    raise NotImplementedError


def max_step_length(s, As, delta):
    """beta = sqrt(2 * delta / (s^T A s)), given s and the product As = A s (Appendix C)."""
    raise NotImplementedError
