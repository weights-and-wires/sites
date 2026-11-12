import numpy as np


def nstep_target(u, v, t, n, gamma):
    """Return z_t = sum_{k<n} gamma^k u[t+k] + gamma^n v[t+n] (no bootstrap past the end)."""
    T = len(u)
    end = min(t + n, T)
    z = 0.0
    for k in range(end - t):
        z += gamma ** k * u[t + k]
    if t + n < T:
        z += gamma ** n * v[t + n]
    return float(z)


def nstep_targets(u, v, n, gamma):
    """Return the array of z_t for t = 0, ..., T - 1."""
    return np.array([nstep_target(u, v, t, n, gamma) for t in range(len(u))])
