import numpy as np


def reverse_source(tokens):
    """Return a new list with the source tokens in reverse order."""
    raise NotImplementedError


def alignment_lags(n, reverse):
    """Return an int array of length n: lag between source token j and target token j
    in the concatenated sequence source + target."""
    raise NotImplementedError
