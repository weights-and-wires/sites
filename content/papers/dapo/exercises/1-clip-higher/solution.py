import numpy as np


def clipped_terms(ratio, adv, eps_low=0.2, eps_high=0.28):
    """Per-token term min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A)."""
    ratio = np.asarray(ratio, dtype=float)
    adv = np.asarray(adv, dtype=float)
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high)
    return np.minimum(ratio * adv, clipped * adv)


def prob_ceiling(p_old, eps_high=0.28):
    """Largest probability a token with A > 0 reaches before its gradient is clipped."""
    return np.minimum(1.0, np.asarray(p_old, dtype=float) * (1 + eps_high))
