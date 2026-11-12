import numpy as np


def ddpg_targets(r, s_next, done, gamma, actor_target, critic_target):
    """Return y of shape (N,): r for terminal rows, r + gamma * Q'(s', mu'(s')) otherwise."""
    a_next = actor_target(s_next)
    q_next = critic_target(s_next, a_next)
    return r + gamma * (1.0 - done) * q_next


def critic_loss(q, y):
    """Return the mean over the batch of (y_i - q_i)**2."""
    return float(np.mean((y - q) ** 2))
