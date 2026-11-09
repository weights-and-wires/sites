import numpy as np


def group_advantages(rewards, eps=1e-6):
    """Return (r - mean) / (std + eps), computed per group over the last axis.

    rewards: array of shape (G,) or (n_questions, G).
    Use the population std (ddof=0). Return a float array of the same shape.
    """
    r = np.asarray(rewards, dtype=float)
    mean = r.mean(axis=-1, keepdims=True)
    std = r.std(axis=-1, keepdims=True)
    return (r - mean) / (std + eps)
