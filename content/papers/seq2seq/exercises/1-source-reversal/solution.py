import numpy as np


def reverse_source(tokens):
    """Return a new list with the source tokens in reverse order."""
    return list(tokens)[::-1]


def alignment_lags(n, reverse):
    """Return an int array of length n: lag between source token j and target token j
    in the concatenated sequence source + target."""
    src_pos = np.arange(n)
    if reverse:
        src_pos = n - 1 - src_pos
    tgt_pos = n + np.arange(n)
    return tgt_pos - src_pos
