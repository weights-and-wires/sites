import numpy as np


def reward_model_loss(rewards, ranking):
    """Return the Equation 1 loss for one prompt as a float.

    rewards: array of shape (K,), the reward-model score of each response.
    ranking: list of K indices into rewards, best response first.
    """
    raise NotImplementedError
