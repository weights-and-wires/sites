import numpy as np


def actor_gradient(theta, S, k):
    """Return (1/N) * sum_i grad_a Q(s_i, a) at a = mu(s_i) times grad_theta mu(s_i), shape (d,).

    mu(s) = theta . s and Q(s, a) = -(a - k . s)**2.
    """
    a = S @ theta
    dq_da = -2.0 * (a - S @ k)
    return np.mean(dq_da[:, None] * S, axis=0)


def ascend(theta, S, k, lr, steps):
    """Take `steps` gradient-ascent steps on J and return the new theta."""
    theta = np.array(theta, dtype=float)
    for _ in range(steps):
        theta = theta + lr * actor_gradient(theta, S, k)
    return theta
