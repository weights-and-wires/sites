import numpy as np


def nstep_target(u, v, t, n, gamma):
    """Return z_t = sum_{k<n} gamma^k u[t+k] + gamma^n v[t+n].

    Stop summing at the end of the episode (len(u)) and add no bootstrap term
    when t + n >= len(u).
    """
    raise NotImplementedError


def nstep_targets(u, v, n, gamma):
    """Return the array of z_t for t = 0, ..., len(u) - 1."""
    raise NotImplementedError
