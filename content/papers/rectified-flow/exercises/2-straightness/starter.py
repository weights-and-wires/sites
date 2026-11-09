import numpy as np


def straightness(path):
    """Discrete straightness S(Z) of Equation 3.

    path has shape (T + 1, n, d): n trajectories sampled at the uniform times
    t_k = k / T. Returns the sum over steps of h * mean over trajectories of
    |(Z1 - Z0) - (Z_{k+1} - Z_k) / h|^2, with h = 1 / T.
    """
    raise NotImplementedError
