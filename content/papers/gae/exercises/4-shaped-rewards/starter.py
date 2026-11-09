import numpy as np


def shape_rewards(rewards, phi, last_phi, gamma):
    """r_tilde_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t) (Equation 20).

    phi[t] is Phi(s_t) for t < T, and last_phi is Phi(s_T).
    """
    raise NotImplementedError


def discounted_cumsum(x, discount):
    """y_t = sum_{l >= 0} discount**l * x[t + l], for every t."""
    raise NotImplementedError


def gae_via_shaping(rewards, values, last_value, gamma, lam):
    """GAE(gamma, lambda) as the (gamma * lambda)-discounted sum of rewards shaped by V (Equation 25)."""
    raise NotImplementedError
