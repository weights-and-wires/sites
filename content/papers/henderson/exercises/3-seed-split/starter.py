import numpy as np


def welch_t_rows(a, b):
    """Row-wise Welch t statistic for two (E, n) matrices. Given: you don't need to change it."""
    na, nb = a.shape[1], b.shape[1]
    va = a.var(axis=1, ddof=1) / na
    vb = b.var(axis=1, ddof=1) / nb
    return (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)


def band_gap_rate(runs):
    """Fraction of rows where the mean +/- 1 SE bands of the two halves do not overlap."""
    raise NotImplementedError


def welch_gap_rate(runs, crit):
    """Fraction of rows where Welch's |t| between the two halves exceeds crit."""
    raise NotImplementedError
