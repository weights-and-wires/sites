import numpy as np


def puct_scores(Q, N, P, c_puct):
    """Return Q + c_puct * P * sqrt(sum(N)) / (1 + N) as an array of shape (K,)."""
    Q = np.asarray(Q, dtype=float)
    N = np.asarray(N, dtype=float)
    P = np.asarray(P, dtype=float)
    return Q + c_puct * P * np.sqrt(N.sum()) / (1.0 + N)


def select_action(Q, N, P, c_puct, legal):
    """Return the index of the highest-scoring legal move (lowest index on ties)."""
    scores = puct_scores(Q, N, P, c_puct)
    scores = np.where(np.asarray(legal, dtype=bool), scores, -np.inf)
    return int(np.argmax(scores))
