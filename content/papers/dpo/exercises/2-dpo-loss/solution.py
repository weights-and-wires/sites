import numpy as np


def reward_margin(pi_w, pi_l, ref_w, ref_l, beta):
    """Return r_hat(x, y_w) - r_hat(x, y_l) for each pair."""
    pi_w, pi_l, ref_w, ref_l = (np.asarray(a, dtype=float) for a in (pi_w, pi_l, ref_w, ref_l))
    return beta * ((pi_w - ref_w) - (pi_l - ref_l))


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    """Return the per-pair DPO loss, shape (B,)."""
    margin = reward_margin(pi_w, pi_l, ref_w, ref_l, beta)
    return np.logaddexp(0.0, -margin)
