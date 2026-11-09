import numpy as np


def kl_closed_form(mu, logvar):
    """KL(N(mu, diag(exp(logvar))) || N(0, I)) per row.

    mu, logvar: arrays of shape (batch, J). Returns an array of shape (batch,).
    """
    raise NotImplementedError


def kl_monte_carlo(mu, logvar, eps):
    """Monte Carlo estimate of the same KL.

    mu, logvar: shape (batch, J). eps: standard normal noise of shape (L, batch, J).
    Returns an array of shape (batch,): the mean over L samples of log q(z) - log p(z),
    where z = mu + exp(0.5 * logvar) * eps.
    """
    raise NotImplementedError
