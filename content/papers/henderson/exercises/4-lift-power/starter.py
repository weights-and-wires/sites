import numpy as np


def welch_t_rows(a, b):
    """Row-wise Welch t statistic for two (B, m) matrices. Given: you don't need to change it."""
    na, nb = a.shape[1], b.shape[1]
    va = a.var(axis=1, ddof=1) / na
    vb = b.var(axis=1, ddof=1) / nb
    return (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)


def lift_power(x, lift, idx_a, idx_b, crit):
    """Fraction of bootstrap pairs (x[idx_a], lift * x[idx_b]) with |Welch t| > crit."""
    raise NotImplementedError
