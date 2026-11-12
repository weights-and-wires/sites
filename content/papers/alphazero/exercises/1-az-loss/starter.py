import numpy as np


def az_loss(logits, v, pi, z, theta, c):
    """The AlphaZero loss l = (z - v)^2 - pi^T log p + c * ||theta||^2.

    logits: raw policy outputs, shape (K,). p is the softmax of logits.
    v: scalar value prediction. pi: search target, shape (K,), sums to 1.
    z: game outcome, scalar. theta: all weights as one array. c: L2 weight.
    Returns a float. Entries of pi that are 0 must contribute 0, even when p
    underflows to 0 for that move.
    """
    raise NotImplementedError


def az_grads(logits, v, pi, z, theta, c):
    """Gradients of az_loss with respect to (logits, v, theta), as a tuple."""
    raise NotImplementedError
