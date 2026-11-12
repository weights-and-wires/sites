import numpy as np


def puct_scores(Q, N, P, c_puct):
    """Return Q + c_puct * P * sqrt(sum(N)) / (1 + N) as an array of shape (K,)."""
    raise NotImplementedError


def select_action(Q, N, P, c_puct, legal):
    """Return the index of the highest-scoring legal move (lowest index on ties)."""
    raise NotImplementedError
