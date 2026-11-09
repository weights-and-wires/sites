import numpy as np

# Appendix D.2, Equation 10.
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss_hat(n, d):
    """Return E + A / n**ALPHA + B / d**BETA for parameters n and tokens d."""
    raise NotImplementedError


def isoflop_curve(c, sizes):
    """For budget c (FLOPs) and model sizes `sizes`, return (d, loss).

    d[i] is the number of tokens that spends exactly c on sizes[i] parameters,
    using C = 6 N D. loss[i] is loss_hat(sizes[i], d[i]).
    """
    raise NotImplementedError
