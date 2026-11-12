import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process: x <- x + theta * (mu - x) * dt + sigma * sqrt(dt) * eps."""

    def __init__(self, dim, theta=0.15, sigma=0.2, mu=0.0, dt=1.0, rng=None):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.dt = dt
        self.rng = rng if rng is not None else np.random.default_rng()
        self.x = np.full(dim, mu, dtype=float)

    def reset(self):
        """Set the state back to mu."""
        self.x = np.full(self.dim, self.mu, dtype=float)

    def sample(self):
        """Take one Euler step and return a copy of the new state, shape (dim,)."""
        eps = self.rng.standard_normal(self.dim)
        self.x = self.x + self.theta * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * eps
        return self.x.copy()
