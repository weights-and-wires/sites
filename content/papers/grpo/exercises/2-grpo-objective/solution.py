import numpy as np


def kl_k3(logp, logp_ref):
    """Equation 4: pi_ref/pi - log(pi_ref/pi) - 1, elementwise, from log-probabilities."""
    d = logp_ref - logp
    return np.exp(d) - d - 1.0


def grpo_objective(logp_new, logp_old, logp_ref, adv, mask, eps, beta):
    """Equation 3 for one group, to be maximized.

    logp_new, logp_old, logp_ref, mask: shape (G, T). adv: shape (G,).
    Per token: min(r * A, clip(r, 1 - eps, 1 + eps) * A) - beta * kl_k3(logp_new, logp_ref).
    Average over each output's own tokens (1/|o_i|), then over the G outputs (1/G).
    """
    mask = np.asarray(mask, dtype=float)
    ratio = np.exp(logp_new - logp_old)
    a = np.asarray(adv, dtype=float)[:, None]
    clipped = np.minimum(ratio * a, np.clip(ratio, 1 - eps, 1 + eps) * a)
    per_token = clipped - beta * kl_k3(logp_new, logp_ref)
    per_output = (per_token * mask).sum(axis=1) / mask.sum(axis=1)
    return per_output.mean()
