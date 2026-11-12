import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process: x <- x + theta * (mu - x) * dt + sigma * sqrt(dt) * eps."""

    def __init__(self, dim, theta=0.15, sigma=0.2, mu=0.0, dt=1.0, rng=None):
        raise NotImplementedError

    def reset(self):
        """Set the state back to mu."""
        raise NotImplementedError

    def sample(self):
        """Take one Euler step and return a copy of the new state, shape (dim,)."""
        raise NotImplementedError
