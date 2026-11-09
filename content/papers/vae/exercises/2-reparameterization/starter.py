import numpy as np


def reparameterize(mu, logvar, eps):
    """Return z = mu + sigma * eps, where logvar = log(sigma^2). Shapes broadcast."""
    raise NotImplementedError


def pathwise_grads(mu, sigma, eps, fprime):
    """Reparameterized gradient estimates of E[f(z)] for z ~ N(mu, sigma^2).

    mu, sigma: floats. eps: 1-D array of standard normal samples.
    fprime: the derivative of f; call it on an array of z values.
    Returns (d_mu, d_sigma), the mean over samples of the per-sample gradients.
    """
    raise NotImplementedError
