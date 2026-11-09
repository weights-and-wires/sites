import numpy as np


def discriminator_loss(real_logits, fake_logits):
    """Equation 1. Each argument is a list of 1-D arrays, one per discriminator k.

    Average max(0, 1 - D(x)) and max(0, 1 + D(G(x))) over time t within each
    discriminator, then over k, and add the two terms.
    """
    raise NotImplementedError


def generator_adv_loss(fake_logits):
    """Equation 2. Average max(0, 1 - D(G(x))) over t, then over k."""
    raise NotImplementedError


def generator_loss(l_adv, l_feat, l_rec, lam_adv=1.0, lam_feat=100.0, lam_rec=1.0):
    """Equation 6: the weighted sum of the three generator losses."""
    raise NotImplementedError
