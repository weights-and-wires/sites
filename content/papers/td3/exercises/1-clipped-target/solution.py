import numpy as np


def clipped_target(r, done, gamma, q1_next, q2_next):
    """Return y of shape (N,): r + gamma * (1 - done) * min(q1_next, q2_next)."""
    q_min = np.minimum(q1_next, q2_next)
    return r + gamma * (1.0 - done) * q_min


def twin_critic_loss(q1, q2, y):
    """Return mean((y - q1)**2) + mean((y - q2)**2)."""
    return float(np.mean((y - q1) ** 2) + np.mean((y - q2) ** 2))
