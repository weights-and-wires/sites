import numpy as np


def add_root_noise(p, legal, eps, alpha, rng):
    """Mix Dirichlet noise into the root prior: P = (1 - eps) * p + eps * eta.

    p: prior over K moves, shape (K,), zero on illegal moves, sums to 1.
    legal: boolean array, shape (K,). eta ~ Dir(alpha) over the legal moves only,
    and eta is 0 on illegal moves. Draw it with one call to
    rng.dirichlet(np.full(m, alpha)), where m is the number of legal moves.
    rng: a numpy Generator. Returns a new array of shape (K,) and leaves p unchanged.
    """
    p = np.asarray(p, dtype=float)
    legal = np.asarray(legal, dtype=bool)
    eta = np.zeros_like(p)
    eta[legal] = rng.dirichlet(np.full(int(legal.sum()), alpha))
    return (1 - eps) * p + eps * eta
