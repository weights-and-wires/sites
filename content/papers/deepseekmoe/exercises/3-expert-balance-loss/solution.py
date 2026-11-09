import numpy as np


def expert_balance_loss(s, idx, alpha=1.0):
    """Return alpha * sum_i f_i * P_i (Equations 12-14)."""
    T, n = s.shape
    k = idx.shape[1]
    counts = np.bincount(idx.ravel(), minlength=n)
    f = counts * n / (k * T)
    p = s.mean(axis=0)
    return float(alpha * (f * p).sum())
