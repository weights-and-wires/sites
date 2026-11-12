import numpy as np


def discounted_state_weights(P, pi, gamma, s0):
    """Return d(s) = sum_t gamma^t Pr(s_t = s | s0, pi), an array of length S."""
    raise NotImplementedError


def action_values(P, R, pi, gamma):
    """Return Q^pi(s, a) as an array of shape (S, A)."""
    raise NotImplementedError


def policy_gradient(P, R, theta, gamma, s0):
    """Return dJ/dtheta, shape (S, A), for the softmax policy with logits theta (Equation 2)."""
    raise NotImplementedError
