import numpy as np


def sample_latent(mu, logvar, eps):
    """Return mu + sigma * eps, where logvar = log(sigma ** 2)."""
    return mu + np.exp(0.5 * logvar) * eps


def kl_to_standard_normal(mu, logvar):
    """Return KL(N(mu, sigma^2) || N(0, 1)) summed over (c, h, w); shape (b,)."""
    kl = 0.5 * (mu**2 + np.exp(logvar) - 1.0 - logvar)
    return kl.sum(axis=(1, 2, 3))


def estimate_scale(z):
    """Return the scalar standard deviation of z over every axis."""
    return float(np.sqrt(np.mean((z - z.mean()) ** 2)))


def rescale(z, sigma_hat):
    """Return z divided by sigma_hat."""
    return z / sigma_hat
