import numpy as np


def kl_categorical(p_old, p_new):
    """Mean over rows of KL[p_old, p_new]. Inputs have shape (N, K)."""
    raise NotImplementedError


def klpen_objective(logp_new, logp_old, adv, p_old, p_new, beta):
    """Return mean(r * A) - beta * mean KL (Equation 8)."""
    raise NotImplementedError


def update_beta(beta, d, d_targ):
    """Adapt the penalty coefficient (rule below Equation 8)."""
    raise NotImplementedError
