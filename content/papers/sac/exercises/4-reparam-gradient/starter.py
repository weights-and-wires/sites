import numpy as np


def reparam_grad(mu, sigma, eps, dq):
    """Return (d_mu, d_sigma) of E[Q(mu + sigma * eps)] from the batch eps, using dq = Q'."""
    raise NotImplementedError


def score_grad(mu, sigma, eps, q):
    """Return (d_mu, d_sigma) of E[Q(a)] with a = mu + sigma * eps, using the score function."""
    raise NotImplementedError
