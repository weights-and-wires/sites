import numpy as np


class ReplayBuffer:
    """A fixed-capacity memory of transitions (s, a, r, s2, done)."""

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.S = np.zeros((capacity, state_dim))
        self.A = np.zeros(capacity, dtype=int)
        self.R = np.zeros(capacity)
        self.S2 = np.zeros((capacity, state_dim))
        self.D = np.zeros(capacity)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, s, a, r, s2, done):
        """Store one transition, overwriting the oldest when full."""
        i = self.pos
        self.S[i], self.A[i], self.R[i], self.S2[i], self.D[i] = s, a, r, s2, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        """Return (S, A, R, S2, D) for batch_size uniform draws with replacement."""
        idx = rng.integers(len(self), size=batch_size)
        return self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.D[idx]
