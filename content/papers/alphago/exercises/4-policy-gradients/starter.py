import numpy as np


def softmax(logits):
    """Return a probability vector of the same shape as logits."""
    raise NotImplementedError


def sl_grad(logits, a):
    """Return d log softmax(logits)[a] / d logits, an array of shape (K,)."""
    raise NotImplementedError


def rl_grad(logits, a, z, baseline=0.0):
    """Return (z - baseline) * d log softmax(logits)[a] / d logits."""
    raise NotImplementedError
