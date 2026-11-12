import numpy as np


def dueling_distribution(v, adv):
    """v: (N,), adv: (A, N). Return (A, N) probabilities: softmax over atoms of v + adv - adv.mean(axis=0)."""
    raise NotImplementedError


def replay_probabilities(priorities, omega):
    """Return priorities**omega normalized to sum to 1."""
    raise NotImplementedError


def importance_weights(probs, beta):
    """Return (N * probs) ** -beta divided by its maximum."""
    raise NotImplementedError
