import numpy as np


def _mean_over_k_of_mean_over_t(per_k):
    return float(np.mean([np.mean(v) for v in per_k]))


def discriminator_loss(real_logits, fake_logits):
    """Equation 1. Each argument is a list of 1-D arrays, one per discriminator k.

    Average max(0, 1 - D(x)) and max(0, 1 + D(G(x))) over time t within each
    discriminator, then over k, and add the two terms.
    """
    real = _mean_over_k_of_mean_over_t([np.maximum(0.0, 1.0 - np.asarray(d, dtype=float)) for d in real_logits])
    fake = _mean_over_k_of_mean_over_t([np.maximum(0.0, 1.0 + np.asarray(d, dtype=float)) for d in fake_logits])
    return real + fake


def generator_adv_loss(fake_logits):
    """Equation 2. Average max(0, 1 - D(G(x))) over t, then over k."""
    return _mean_over_k_of_mean_over_t([np.maximum(0.0, 1.0 - np.asarray(d, dtype=float)) for d in fake_logits])


def generator_loss(l_adv, l_feat, l_rec, lam_adv=1.0, lam_feat=100.0, lam_rec=1.0):
    """Equation 6: the weighted sum of the three generator losses."""
    return lam_adv * l_adv + lam_feat * l_feat + lam_rec * l_rec
