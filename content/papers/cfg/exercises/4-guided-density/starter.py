import numpy as np


def guided_density(log_pc, log_p, w, dx):
    """Normalized guided density on a grid (Section 3.1, Section 3.2):

    p_tilde(x | c) proportional to p(x | c)^(1 + w) * p(x)^(-w).

    log_pc and log_p are arrays of log densities on a grid with spacing dx.
    Return an array that sums to 1 when multiplied by dx. It must not overflow
    when the log densities are large.
    """
    raise NotImplementedError


def density_stats(x, p, dx):
    """Mean and standard deviation of the density p on the grid x."""
    raise NotImplementedError
