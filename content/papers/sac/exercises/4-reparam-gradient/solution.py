import numpy as np


def reparam_grad(mu, sigma, eps, dq):
    """Return (d_mu, d_sigma) of E[Q(mu + sigma * eps)] from the batch eps, using dq = Q'."""
    eps = np.asarray(eps, dtype=float)
    g = dq(mu + sigma * eps)
    return float(np.mean(g)), float(np.mean(g * eps))


def score_grad(mu, sigma, eps, q):
    """Return (d_mu, d_sigma) of E[Q(a)] with a = mu + sigma * eps, using the score function."""
    eps = np.asarray(eps, dtype=float)
    a = mu + sigma * eps
    value = q(a)
    score_mu = (a - mu) / sigma**2
    score_sigma = (a - mu) ** 2 / sigma**3 - 1.0 / sigma
    return float(np.mean(value * score_mu)), float(np.mean(value * score_sigma))
