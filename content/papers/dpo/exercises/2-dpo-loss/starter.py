import numpy as np


def reward_margin(pi_w, pi_l, ref_w, ref_l, beta):
    """Return r_hat(x, y_w) - r_hat(x, y_l) for each pair.

    All four inputs are arrays of sequence log-probabilities, shape (B,).
    r_hat = beta * (log pi_theta - log pi_ref).
    """
    raise NotImplementedError


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    """Return the per-pair DPO loss, shape (B,). Must stay finite for margins of +-1000."""
    raise NotImplementedError
