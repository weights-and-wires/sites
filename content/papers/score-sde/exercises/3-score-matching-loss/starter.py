import numpy as np

BMIN, BMAX = 0.1, 20.0


def int_beta(t):
    """Finished for you. Integral of beta(s) = BMIN + s (BMAX - BMIN) from 0 to t."""
    return BMIN * t + 0.5 * (BMAX - BMIN) * t**2


def dsm_loss(score_model, x0, t, eps):
    """One evaluation of Equation 7 with lambda(t) = sigma^2(t).

    score_model(x_t, t) returns an array shaped like x_t.
    x0, eps: (batch, dim). t: (batch,) floats in (0, 1].
    Returns a float: summed over dim, averaged over the batch.
    """
    raise NotImplementedError
