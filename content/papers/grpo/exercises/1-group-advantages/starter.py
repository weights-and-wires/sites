import numpy as np


def group_advantages(rewards, eps=1e-6):
    """Return (r - mean) / (std + eps), computed per group over the last axis.

    rewards: array of shape (G,) or (n_questions, G).
    Use the population std (ddof=0). Return a float array of the same shape.
    """
    raise NotImplementedError
