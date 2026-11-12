import numpy as np


def clipped_terms(ratio, adv, eps_low=0.2, eps_high=0.28):
    """Per-token term min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A)."""
    raise NotImplementedError


def prob_ceiling(p_old, eps_high=0.28):
    """Largest probability a token with A > 0 reaches before its gradient is clipped."""
    raise NotImplementedError
