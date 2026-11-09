import numpy as np


def shape_rewards(rewards, phi, last_phi, gamma):
    """r_tilde_t = r_t + gamma * Phi(s_{t+1}) - Phi(s_t) (Equation 20).

    phi[t] is Phi(s_t) for t < T, and last_phi is Phi(s_T).
    """
    rewards = np.asarray(rewards, dtype=float)
    phi = np.asarray(phi, dtype=float)
    next_phi = np.append(phi[1:], last_phi)
    return rewards + gamma * next_phi - phi


def discounted_cumsum(x, discount):
    """y_t = sum_{l >= 0} discount**l * x[t + l], for every t."""
    x = np.asarray(x, dtype=float)
    y = np.zeros(len(x))
    acc = 0.0
    for t in reversed(range(len(x))):
        acc = x[t] + discount * acc
        y[t] = acc
    return y


def gae_via_shaping(rewards, values, last_value, gamma, lam):
    """GAE(gamma, lambda) as the (gamma * lambda)-discounted sum of rewards shaped by V (Equation 25)."""
    shaped = shape_rewards(rewards, values, last_value, gamma)
    return discounted_cumsum(shaped, gamma * lam)
