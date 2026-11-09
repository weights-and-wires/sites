import numpy as np


def reward_model_loss(rewards, ranking):
    """Return the Equation 1 loss for one prompt as a float.

    rewards: array of shape (K,), the reward-model score of each response.
    ranking: list of K indices into rewards, best response first.
    """
    rewards = np.asarray(rewards, dtype=float)
    losses = []
    for i in range(len(ranking)):
        for j in range(i + 1, len(ranking)):
            diff = rewards[ranking[i]] - rewards[ranking[j]]
            losses.append(np.logaddexp(0.0, -diff))
    return float(np.mean(losses))
