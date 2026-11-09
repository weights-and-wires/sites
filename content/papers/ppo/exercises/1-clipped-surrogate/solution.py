import numpy as np


def clipped_surrogate(logp_new, logp_old, adv, eps=0.2):
    """Return the scalar mean of min(r*A, clip(r, 1-eps, 1+eps)*A) (Equation 7)."""
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    adv = np.asarray(adv)
    return float(np.mean(np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)))


def clip_grad_logp(logp_new, logp_old, adv, eps=0.2):
    """Return d(clipped_surrogate)/d(logp_new), an array of length N."""
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    adv = np.asarray(adv)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv
    active = unclipped <= clipped
    return np.where(active, unclipped, 0.0) / len(ratio)
