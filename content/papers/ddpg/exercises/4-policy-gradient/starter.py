import numpy as np


def actor_gradient(theta, S, k):
    """Return (1/N) * sum_i grad_a Q(s_i, a) at a = mu(s_i) times grad_theta mu(s_i), shape (d,).

    mu(s) = theta . s and Q(s, a) = -(a - k . s)**2.
    """
    raise NotImplementedError


def ascend(theta, S, k, lr, steps):
    """Take `steps` gradient-ascent steps on J and return the new theta."""
    raise NotImplementedError
