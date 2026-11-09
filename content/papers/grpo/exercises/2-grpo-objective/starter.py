import numpy as np


def kl_k3(logp, logp_ref):
    """Equation 4: pi_ref/pi - log(pi_ref/pi) - 1, elementwise, from log-probabilities."""
    raise NotImplementedError


def grpo_objective(logp_new, logp_old, logp_ref, adv, mask, eps, beta):
    """Equation 3 for one group, to be maximized.

    logp_new, logp_old, logp_ref, mask: shape (G, T). adv: shape (G,).
    Per token: min(r * A, clip(r, 1 - eps, 1 + eps) * A) - beta * kl_k3(logp_new, logp_ref).
    Average over each output's own tokens (1/|o_i|), then over the G outputs (1/G).
    """
    raise NotImplementedError
