import numpy as np


def _softmax_rows(theta):
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def discounted_state_weights(P, pi, gamma, s0):
    """Return d(s) = sum_t gamma^t Pr(s_t = s | s0, pi), an array of length S."""
    p_pi = np.einsum("sa,sat->st", pi, P)
    e0 = np.zeros(P.shape[0])
    e0[s0] = 1.0
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * p_pi.T, e0)


def action_values(P, R, pi, gamma):
    """Return Q^pi(s, a) as an array of shape (S, A)."""
    p_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    v = np.linalg.solve(np.eye(P.shape[0]) - gamma * p_pi, r_pi)
    return R + gamma * np.einsum("sat,t->sa", P, v)


def policy_gradient(P, R, theta, gamma, s0):
    """Return dJ/dtheta, shape (S, A), for the softmax policy with logits theta (Equation 2)."""
    pi = _softmax_rows(theta)
    q = action_values(P, R, pi, gamma)
    d = discounted_state_weights(P, pi, gamma, s0)
    v = (pi * q).sum(axis=1, keepdims=True)
    return d[:, None] * pi * (q - v)
