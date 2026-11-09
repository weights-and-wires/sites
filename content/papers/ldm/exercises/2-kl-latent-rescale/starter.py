import numpy as np


def sample_latent(mu, logvar, eps):
    """Return mu + sigma * eps, where logvar = log(sigma ** 2)."""
    raise NotImplementedError


def kl_to_standard_normal(mu, logvar):
    """Return KL(N(mu, sigma^2) || N(0, 1)) summed over (c, h, w); shape (b,)."""
    raise NotImplementedError


def estimate_scale(z):
    """Return the scalar standard deviation of z over every axis."""
    raise NotImplementedError


def rescale(z, sigma_hat):
    """Return z divided by sigma_hat."""
    raise NotImplementedError
