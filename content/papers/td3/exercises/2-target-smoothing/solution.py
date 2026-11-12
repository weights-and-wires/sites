import numpy as np


def smoothed_action(mu_next, z, sigma, c, low, high):
    """Return clip(mu_next + clip(sigma * z, -c, c), low, high)."""
    noise = np.clip(sigma * z, -c, c)
    return np.clip(mu_next + noise, low, high)


def sample_smoothed_action(mu_next, sigma, c, low, high, rng):
    """Draw z = rng.standard_normal(mu_next.shape) once and return smoothed_action(...)."""
    z = rng.standard_normal(mu_next.shape)
    return smoothed_action(mu_next, z, sigma, c, low, high)
