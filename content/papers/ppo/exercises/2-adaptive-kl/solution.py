import numpy as np


def kl_categorical(p_old, p_new):
    """Mean over rows of KL[p_old, p_new]. Inputs have shape (N, K)."""
    p_old = np.atleast_2d(np.asarray(p_old, dtype=float))
    p_new = np.atleast_2d(np.asarray(p_new, dtype=float))
    return float(np.mean(np.sum(p_old * (np.log(p_old) - np.log(p_new)), axis=-1)))


def klpen_objective(logp_new, logp_old, adv, p_old, p_new, beta):
    """Return mean(r * A) - beta * mean KL (Equation 8)."""
    ratio = np.exp(np.asarray(logp_new) - np.asarray(logp_old))
    return float(np.mean(ratio * np.asarray(adv)) - beta * kl_categorical(p_old, p_new))


def update_beta(beta, d, d_targ):
    """Adapt the penalty coefficient (rule below Equation 8)."""
    if d < d_targ / 1.5:
        return beta / 2
    if d > d_targ * 1.5:
        return beta * 2
    return beta
