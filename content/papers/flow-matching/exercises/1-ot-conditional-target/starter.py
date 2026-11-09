import numpy as np


def ot_pair(x0, x1, t, sigma_min):
    """Return (xt, ut) for the OT conditional path (Equations 22 and 23).

    x0, x1: arrays of shape (n, d). t: array of shape (n,).
    """
    raise NotImplementedError


def cfm_loss(v, x0, x1, t, sigma_min):
    """Mean over the batch of ||v(xt, t) - ut||^2 (Equation 23)."""
    raise NotImplementedError
