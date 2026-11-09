import numpy as np


def td_targets(r, q_next, done, gamma):
    """Return y of shape (B,): r for terminal rows, r + gamma * max_a' q_next otherwise."""
    raise NotImplementedError


def dqn_loss(q, a, y):
    """Return the mean over the batch of (y_j - q[j, a_j])**2."""
    raise NotImplementedError
