import numpy as np


def returns_to_go(rewards):
    """Undiscounted return-to-go along the last axis.

    rewards: array-like of shape (T,) or (B, T).
    Returns a float array of the same shape; out[..., t] = sum(rewards[..., t:]).
    Does not modify its input.
    """
    r = np.asarray(rewards, dtype=float)
    return np.cumsum(r[..., ::-1], axis=-1)[..., ::-1]
