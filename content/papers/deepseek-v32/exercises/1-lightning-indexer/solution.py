import numpy as np


def index_scores(Q, w, K):
    """Return the (L, L) lightning indexer scores of Equation 1, with -inf for s > t."""
    L = K.shape[0]
    dots = np.einsum("tjd,sd->tjs", Q, K)
    raw = np.einsum("tj,tjs->ts", w, np.maximum(dots, 0.0))
    return np.where(np.tril(np.ones((L, L), dtype=bool)), raw, -np.inf)
