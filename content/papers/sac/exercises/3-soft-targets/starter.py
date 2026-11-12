import numpy as np


def v_target(q1, q2, logp):
    """Return min(q1, q2) - logp, shape (N,)."""
    raise NotImplementedError


def q_target(r, v_next, done, gamma):
    """Return r + gamma * (1 - done) * v_next, shape (N,)."""
    raise NotImplementedError


def half_mse(pred, target):
    """Return the mean over the batch of 0.5 * (pred - target)**2."""
    raise NotImplementedError
