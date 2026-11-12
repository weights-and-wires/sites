import numpy as np


def _log_softmax(x):
    x = np.asarray(x, dtype=float)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def az_loss(logits, v, pi, z, theta, c):
    """The AlphaZero loss l = (z - v)^2 - pi^T log p + c * ||theta||^2.

    logits: raw policy outputs, shape (K,). p is the softmax of logits.
    v: scalar value prediction. pi: search target, shape (K,), sums to 1.
    z: game outcome, scalar. theta: all weights as one array. c: L2 weight.
    Returns a float. Entries of pi that are 0 must contribute 0, even when p
    underflows to 0 for that move.
    """
    pi = np.asarray(pi, dtype=float)
    logp = _log_softmax(logits)
    value_term = (z - v) ** 2
    policy_term = -float(np.sum(pi * logp))
    l2 = c * float(np.sum(np.asarray(theta, dtype=float) ** 2))
    return value_term + policy_term + l2


def az_grads(logits, v, pi, z, theta, c):
    """Gradients of az_loss with respect to (logits, v, theta), as a tuple."""
    p = np.exp(_log_softmax(logits))
    d_logits = p - np.asarray(pi, dtype=float)
    d_v = -2.0 * (z - v)
    d_theta = 2.0 * c * np.asarray(theta, dtype=float)
    return d_logits, d_v, d_theta
