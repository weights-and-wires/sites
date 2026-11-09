import numpy as np


def log_px(x):
    """Finished for you: log p(x) for p(x) = N(x; 0, 2), the marginal of this model."""
    return -0.5 * np.log(2 * np.pi * 2.0) - x**2 / 4.0


def elbo_estimate(x, mu, logvar, eps):
    """SGVB estimator B (Equation 7) for p(z) = N(0, 1), p(x|z) = N(x; z, 1).

    x, mu, logvar: floats. eps: 1-D array of L standard normal samples.
    Returns -KL(q || p(z)) + mean over l of log p(x | z_l), with z_l = mu + sigma * eps_l.
    """
    sigma = np.exp(0.5 * logvar)
    z = mu + sigma * eps
    neg_kl = 0.5 * (1 + logvar - mu**2 - np.exp(logvar))
    log_lik = -0.5 * np.log(2 * np.pi) - 0.5 * (x - z) ** 2
    return neg_kl + np.mean(log_lik)
