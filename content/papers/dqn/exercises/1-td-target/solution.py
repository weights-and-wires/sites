import numpy as np


def td_targets(r, q_next, done, gamma):
    """Return y of shape (B,): r for terminal rows, r + gamma * max_a' q_next otherwise."""
    return r + gamma * (1.0 - done) * np.max(q_next, axis=1)


def dqn_loss(q, a, y):
    """Return the mean over the batch of (y_j - q[j, a_j])**2."""
    q_taken = q[np.arange(len(a)), a]
    return float(np.mean((y - q_taken) ** 2))
