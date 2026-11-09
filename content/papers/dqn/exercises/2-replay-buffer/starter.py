import numpy as np


class ReplayBuffer:
    """A fixed-capacity memory of transitions (s, a, r, s2, done)."""

    def __init__(self, capacity, state_dim):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def push(self, s, a, r, s2, done):
        """Store one transition, overwriting the oldest when full."""
        raise NotImplementedError

    def sample(self, batch_size, rng):
        """Return (S, A, R, S2, D) for batch_size uniform draws with replacement."""
        raise NotImplementedError
