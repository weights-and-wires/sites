import numpy as np


def straightness(path):
    """Discrete straightness S(Z) of Equation 3.

    path has shape (T + 1, n, d): n trajectories sampled at the uniform times
    t_k = k / T. Returns the sum over steps of h * mean over trajectories of
    |(Z1 - Z0) - (Z_{k+1} - Z_k) / h|^2, with h = 1 / T.
    """
    path = np.asarray(path, dtype=float)
    steps = len(path) - 1
    h = 1.0 / steps
    chord = path[-1] - path[0]
    rate = (path[1:] - path[:-1]) / h
    per_step = ((chord[None] - rate) ** 2).sum(axis=-1).mean(axis=1)
    return float(h * per_step.sum())
