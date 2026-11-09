import numpy as np


def l2_normalize(x, axis=-1):
    """Scale each vector along axis to unit length."""
    raise NotImplementedError


def clip_loss(I_f, T_f, t):
    """Return the symmetric contrastive loss of Figure 3 as a float.

    I_f, T_f: (n, d_e) features of n aligned (image, text) pairs, not normalized.
    t: log of the logit scale, so the scale is exp(t).
    """
    raise NotImplementedError
