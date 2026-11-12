import numpy as np


def dueling_distribution(v, adv):
    """v: (N,), adv: (A, N). Return (A, N) probabilities: softmax over atoms of v + adv - adv.mean(axis=0)."""
    v = np.asarray(v, dtype=float)
    adv = np.asarray(adv, dtype=float)
    logits = v[None, :] + adv - adv.mean(axis=0, keepdims=True)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def replay_probabilities(priorities, omega):
    """Return priorities**omega normalized to sum to 1."""
    p = np.asarray(priorities, dtype=float) ** omega
    return p / p.sum()


def importance_weights(probs, beta):
    """Return (N * probs) ** -beta divided by its maximum."""
    probs = np.asarray(probs, dtype=float)
    w = (len(probs) * probs) ** (-beta)
    return w / w.max()
