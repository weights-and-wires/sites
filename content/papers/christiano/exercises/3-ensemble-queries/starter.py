import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def segment_returns(F, w):
    """Finished for you: the return of each segment under weights w. F has shape (n, k, d)."""
    return F.sum(axis=1) @ w


def normalize_rewards(r):
    """Zero mean and unit standard deviation over all entries. A constant input gives zeros."""
    raise NotImplementedError


def ensemble_reward(ws, F):
    """Average of the independently normalized per-step rewards F @ w. Shape (n, k)."""
    raise NotImplementedError


def select_queries(ws, F1, F2, m):
    """Indices of the m pairs with the highest variance of P_hat[sigma1 > sigma2] across members."""
    raise NotImplementedError
