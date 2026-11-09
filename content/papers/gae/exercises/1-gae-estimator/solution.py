import numpy as np


def td_residuals(rewards, values, last_value, gamma):
    """delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) for t = 0..T-1.

    values[t] is V(s_t), and last_value is V(s_T), the bootstrap value after the last step.
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    next_values = np.append(values[1:], last_value)
    return rewards + gamma * next_values - values


def gae(rewards, values, last_value, gamma, lam):
    """GAE(gamma, lambda) advantages for every step: sum_l (gamma * lam)^l delta_{t+l} (Equation 16)."""
    delta = td_residuals(rewards, values, last_value, gamma)
    adv = np.zeros(len(delta))
    acc = 0.0
    for t in reversed(range(len(delta))):
        acc = delta[t] + gamma * lam * acc
        adv[t] = acc
    return adv
