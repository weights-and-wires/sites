import numpy as np


def make_infill_inputs(x1, x0, t, mask):
    """Return (x_t, cond, target) for one training example.

    x1, x0: (N, F) data and noise. t: float in [0, 1]. mask: (N,) of 0/1,
    where 1 marks a frame the model generates.
    x_t = (1 - t) * x0 + t * x1, cond = (1 - mask) * x1, target = x1 - x0.
    Do not modify the inputs.
    """
    m = np.asarray(mask, dtype=float)[:, None]
    x_t = (1 - t) * x0 + t * x1
    cond = (1 - m) * x1
    target = x1 - x0
    return x_t, cond, target


def masked_cfm_loss(pred, target, mask):
    """Mean squared error over the masked frames only (a float).

    Divide the sum of the masked squared errors by mask.sum() * F.
    """
    m = np.asarray(mask, dtype=float)[:, None]
    err = (pred - target) ** 2 * m
    return float(err.sum() / (m.sum() * pred.shape[1]))
