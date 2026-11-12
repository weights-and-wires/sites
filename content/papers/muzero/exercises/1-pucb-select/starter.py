import numpy as np


def minmax_normalize(Q, lo, hi):
    """Return (Q - lo) / (hi - lo) as a float array, or zeros when hi <= lo."""
    raise NotImplementedError


def puct_scores(Q, N, P, lo, hi, c1=1.25, c2=19652.0):
    """Return the pUCT score of every edge, as an array of shape (K,)."""
    raise NotImplementedError


def select_action(Q, N, P, lo, hi, legal=None, c1=1.25, c2=19652.0):
    """Return the index of the highest-scoring allowed edge (lowest index on ties)."""
    raise NotImplementedError
