import numpy as np


def gae_rollout(rewards, values, dones, last_value, gamma, lam):
    """GAE(gamma, lambda) over a flat rollout buffer that can hold several episodes.

    dones[t] = 1 means the episode ended right after step t. last_value is V(s_N), used
    only when the last step is not a done step.
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=float)
    n = len(rewards)
    adv = np.zeros(n)
    acc = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        acc = delta + gamma * lam * mask * acc
        adv[t] = acc
    return adv
