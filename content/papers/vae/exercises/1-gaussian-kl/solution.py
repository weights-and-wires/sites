import numpy as np


def kl_closed_form(mu, logvar):
    """KL(N(mu, diag(exp(logvar))) || N(0, I)) per row.

    mu, logvar: arrays of shape (batch, J). Returns an array of shape (batch,).
    """
    return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar), axis=-1)


def kl_monte_carlo(mu, logvar, eps):
    """Monte Carlo estimate of the same KL.

    mu, logvar: shape (batch, J). eps: standard normal noise of shape (L, batch, J).
    Returns an array of shape (batch,): the mean over L samples of log q(z) - log p(z),
    where z = mu + exp(0.5 * logvar) * eps.
    """
    sigma = np.exp(0.5 * logvar)
    z = mu + sigma * eps
    log_q = -0.5 * np.sum(np.log(2 * np.pi) + logvar + eps**2, axis=-1)
    log_p = -0.5 * np.sum(np.log(2 * np.pi) + z**2, axis=-1)
    return np.mean(log_q - log_p, axis=0)
