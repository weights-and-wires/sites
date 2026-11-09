import numpy as np


def guided_density(log_pc, log_p, w, dx):
    """Normalized guided density on a grid (Section 3.1, Section 3.2):

    p_tilde(x | c) proportional to p(x | c)^(1 + w) * p(x)^(-w).

    log_pc and log_p are arrays of log densities on a grid with spacing dx.
    Return an array that sums to 1 when multiplied by dx. It must not overflow
    when the log densities are large.
    """
    logits = (1.0 + w) * np.asarray(log_pc) - w * np.asarray(log_p)
    logits = logits - logits.max()
    p = np.exp(logits)
    return p / (p.sum() * dx)


def density_stats(x, p, dx):
    """Mean and standard deviation of the density p on the grid x."""
    mean = np.sum(x * p) * dx
    var = np.sum((x - mean) ** 2 * p) * dx
    return mean, np.sqrt(var)
