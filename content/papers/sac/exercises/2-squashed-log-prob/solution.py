import numpy as np


def squashed_log_prob(u, mu, std):
    """Return log pi(tanh(u)) for u ~ N(mu, std^2), summed over the last axis."""
    u = np.asarray(u, dtype=float)
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    log_normal = -0.5 * ((u - mu) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)
    log_one_minus_tanh_sq = 2.0 * (np.log(2.0) - u - np.logaddexp(0.0, -2.0 * u))
    return np.sum(log_normal - log_one_minus_tanh_sq, axis=-1)
