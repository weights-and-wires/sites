import numpy as np


def gae_rollout(rewards, values, dones, last_value, gamma, lam):
    """GAE(gamma, lambda) over a flat rollout buffer that can hold several episodes.

    dones[t] = 1 means the episode ended right after step t. last_value is V(s_N), used
    only when the last step is not a done step.
    """
    raise NotImplementedError
