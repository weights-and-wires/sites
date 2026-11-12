import numpy as np


def v_target(q1, q2, logp):
    """Return min(q1, q2) - logp, shape (N,)."""
    return np.minimum(q1, q2) - logp


def q_target(r, v_next, done, gamma):
    """Return r + gamma * (1 - done) * v_next, shape (N,)."""
    return r + gamma * (1.0 - done) * v_next


def half_mse(pred, target):
    """Return the mean over the batch of 0.5 * (pred - target)**2."""
    return float(np.mean(0.5 * (pred - target) ** 2))
