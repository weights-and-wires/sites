import numpy as np

# Appendix K, Table 2 (alpha = beta).
A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965


def p_star(gamma_bar):
    """Return the compute-optimal precision P* = x * gamma_bar.

    x is the positive root of exp(x) - 1 = 3 x, found by bisection on [1, 10].
    """
    raise NotImplementedError


def best_bits(c, bits, gamma_bar):
    """Return (p, loss): the precision in `bits` with the lowest loss at budget c.

    For each p, use the cost C = (6/16) N D p, so D = c / (6/16 * N * p), and
    the loss A (N (1 - exp(-p/gamma_bar))**3)**-ALPHA + B D**-ALPHA + E.
    Take the lowest loss over a grid of N, then the lowest over `bits`.
    """
    raise NotImplementedError
