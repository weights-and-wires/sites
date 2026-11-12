import numpy as np


def learnability_reward(solver_rewards):
    """Return the proposer's learnability reward (Equation 4).

    solver_rewards: 1-D sequence of n solver rewards, each 0 or 1.
    Return 0.0 if the mean is 0, else 1 - mean, as a float.
    """
    mean = float(np.mean(np.asarray(solver_rewards, dtype=float)))
    if mean == 0.0:
        return 0.0
    return 1.0 - mean
