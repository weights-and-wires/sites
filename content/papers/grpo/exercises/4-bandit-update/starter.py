import numpy as np


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def group_advantages(rewards, eps=1e-6):
    """Finished for you (Exercise 1): (r - mean) / (std + eps) over the group."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean(axis=-1, keepdims=True)) / (r.std(axis=-1, keepdims=True) + eps)


def grpo_update(logits, actions, rewards, logits_ref, lr, beta):
    """One gradient-ascent step on Equation 3 for a one-token softmax policy.

    logits, logits_ref: shape (K,). actions: shape (G,), ints. rewards: shape (G,).
    Return the new logits as a new array.
    """
    raise NotImplementedError
