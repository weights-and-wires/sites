import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def segment_returns(F, w):
    """Finished for you: the return of each segment under weights w. F has shape (n, k, d)."""
    return F.sum(axis=1) @ w


def normalize_rewards(r):
    """Zero mean and unit standard deviation over all entries. A constant input gives zeros."""
    r = np.asarray(r, dtype=float)
    sd = r.std()
    if sd == 0:
        return np.zeros_like(r)
    return (r - r.mean()) / sd


def ensemble_reward(ws, F):
    """Average of the independently normalized per-step rewards F @ w. Shape (n, k)."""
    return np.mean([normalize_rewards(F @ w) for w in ws], axis=0)


def select_queries(ws, F1, F2, m):
    """Indices of the m pairs with the highest variance of P_hat[sigma1 > sigma2] across members."""
    probs = np.array([sigmoid(segment_returns(F1, w) - segment_returns(F2, w)) for w in ws])
    var = probs.var(axis=0)
    return np.argsort(-var, kind="stable")[:m]
