import numpy as np


def welch_t_rows(a, b):
    """Row-wise Welch t statistic for two (E, n) matrices. Given: you don't need to change it."""
    na, nb = a.shape[1], b.shape[1]
    va = a.var(axis=1, ddof=1) / na
    vb = b.var(axis=1, ddof=1) / nb
    return (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)


def band_gap_rate(runs):
    """Fraction of rows where the mean +/- 1 SE bands of the two halves do not overlap."""
    runs = np.asarray(runs, dtype=float)
    n = runs.shape[1] // 2
    a, b = runs[:, :n], runs[:, n:]
    se_a = a.std(axis=1, ddof=1) / np.sqrt(n)
    se_b = b.std(axis=1, ddof=1) / np.sqrt(n)
    return float(np.mean(np.abs(a.mean(axis=1) - b.mean(axis=1)) > se_a + se_b))


def welch_gap_rate(runs, crit):
    """Fraction of rows where Welch's |t| between the two halves exceeds crit."""
    runs = np.asarray(runs, dtype=float)
    n = runs.shape[1] // 2
    t = welch_t_rows(runs[:, :n], runs[:, n:])
    return float(np.mean(np.abs(t) > crit))
