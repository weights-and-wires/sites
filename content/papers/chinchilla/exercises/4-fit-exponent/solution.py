import numpy as np


def fit_exponent(c, y):
    """Return p in y = k * c**p from a least-squares line through (ln c, ln y)."""
    slope, _ = np.polyfit(np.log(np.asarray(c, dtype=float)), np.log(np.asarray(y, dtype=float)), 1)
    return float(slope)


def grow(n0, d0, k, a):
    """Spend k times the compute of a run with n0 parameters and d0 tokens.

    Return (n1, d1) with n1 = n0 * k**a and d1 = d0 * k**(1 - a).
    """
    return n0 * k**a, d0 * k ** (1 - a)
