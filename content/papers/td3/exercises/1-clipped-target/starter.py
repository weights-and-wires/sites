import numpy as np


def clipped_target(r, done, gamma, q1_next, q2_next):
    """Return y of shape (N,): r + gamma * (1 - done) * min(q1_next, q2_next)."""
    raise NotImplementedError


def twin_critic_loss(q1, q2, y):
    """Return mean((y - q1)**2) + mean((y - q2)**2)."""
    raise NotImplementedError
