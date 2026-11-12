import numpy as np


def ddpg_targets(r, s_next, done, gamma, actor_target, critic_target):
    """Return y of shape (N,): r for terminal rows, r + gamma * Q'(s', mu'(s')) otherwise."""
    raise NotImplementedError


def critic_loss(q, y):
    """Return the mean over the batch of (y_i - q_i)**2."""
    raise NotImplementedError
