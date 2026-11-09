import numpy as np


def td_residuals(rewards, values, last_value, gamma):
    """delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) for t = 0..T-1.

    values[t] is V(s_t), and last_value is V(s_T), the bootstrap value after the last step.
    """
    raise NotImplementedError


def gae(rewards, values, last_value, gamma, lam):
    """GAE(gamma, lambda) advantages for every step: sum_l (gamma * lam)^l delta_{t+l} (Equation 16)."""
    raise NotImplementedError
