import numpy as np


def alpha_sigma(lam):
    """Return (alpha, sigma) for log-SNR lam, with alpha^2 = 1 / (1 + e^-lam)
    and sigma^2 = 1 - alpha^2 (Section 2, Equation 1). lam can be an array."""
    raise NotImplementedError


def sample_lambda(u, lam_min=-20.0, lam_max=20.0):
    """Map u in [0, 1] to a log-SNR with lam = -2 log tan(a u + b) (Section 2).

    b = arctan(exp(-lam_max / 2)) and a = arctan(exp(-lam_min / 2)) - b.
    """
    raise NotImplementedError


def drop_condition(c, p_uncond, rng, null=-1):
    """Algorithm 1, line 3: replace each label by `null` with probability p_uncond.

    c is an integer array of shape (batch,). Return a new array and leave c alone.
    """
    raise NotImplementedError


def corrupt(x, lam, eps):
    """Algorithm 1, line 6: z = alpha_lam * x + sigma_lam * eps.

    x and eps have shape (batch, dim). lam is a float or an array of shape (batch,).
    """
    raise NotImplementedError
