import numpy as np


def clipped_surrogate(logp_new, logp_old, adv, eps=0.2):
    """Return the scalar mean of min(r*A, clip(r, 1-eps, 1+eps)*A) (Equation 7)."""
    raise NotImplementedError


def clip_grad_logp(logp_new, logp_old, adv, eps=0.2):
    """Return d(clipped_surrogate)/d(logp_new), an array of length N."""
    raise NotImplementedError
