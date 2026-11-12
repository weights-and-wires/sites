import numpy as np


def minmax_normalize(Q, lo, hi):
    """Return (Q - lo) / (hi - lo), or zeros when hi <= lo."""
    Q = np.asarray(Q, dtype=float)
    if hi <= lo:
        return np.zeros_like(Q)
    return (Q - lo) / (hi - lo)


def puct_scores(Q, N, P, lo, hi, c1=1.25, c2=19652.0):
    """Return the pUCT score of every edge, as an array of shape (K,)."""
    N = np.asarray(N, dtype=float)
    P = np.asarray(P, dtype=float)
    total = N.sum()
    q = minmax_normalize(Q, lo, hi)
    bonus = P * np.sqrt(total) / (1.0 + N) * (c1 + np.log((total + c2 + 1.0) / c2))
    return q + bonus


def select_action(Q, N, P, lo, hi, legal=None, c1=1.25, c2=19652.0):
    """Return the index of the highest-scoring allowed edge (lowest index on ties)."""
    scores = puct_scores(Q, N, P, lo, hi, c1, c2)
    if legal is not None:
        scores = np.where(np.asarray(legal, dtype=bool), scores, -np.inf)
    return int(np.argmax(scores))
