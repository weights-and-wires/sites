import numpy as np


def make_adding_sequence(rng, T):
    """Return (X, target) for one adding-problem sequence (Section 5.4.1).

    X has shape (L, 2) with L drawn uniformly from T..T + T // 10. Column 0 holds
    values from [-1, 1]. Column 1 holds markers: 1.0 for the two marked pairs,
    -1.0 for the first and last pairs, and 0.0 elsewhere.
    """
    raise NotImplementedError


def cec_adder(X):
    """Add the marked values with an idealized memory cell and return the target."""
    raise NotImplementedError
