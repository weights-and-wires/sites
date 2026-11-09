import numpy as np


def ot_pair(x0, x1, t, sigma_min):
    """Return (xt, ut) for the OT conditional path (Equations 22 and 23).

    x0, x1: arrays of shape (n, d). t: array of shape (n,).
    """
    t = np.asarray(t, dtype=float)[:, None]
    xt = (1.0 - (1.0 - sigma_min) * t) * x0 + t * x1
    ut = x1 - (1.0 - sigma_min) * x0
    return xt, ut


def cfm_loss(v, x0, x1, t, sigma_min):
    """Mean over the batch of ||v(xt, t) - ut||^2 (Equation 23)."""
    xt, ut = ot_pair(x0, x1, t, sigma_min)
    diff = v(xt, t) - ut
    return float(np.mean(np.sum(diff**2, axis=-1)))
