import numpy as np


def alpha_sigma(lam):
    """Return (alpha, sigma) for log-SNR lam, with alpha^2 = 1 / (1 + e^-lam)
    and sigma^2 = 1 - alpha^2 (Section 2, Equation 1). lam can be an array."""
    lam = np.asarray(lam, dtype=float)
    alpha_sq = 1.0 / (1.0 + np.exp(-lam))
    return np.sqrt(alpha_sq), np.sqrt(1.0 - alpha_sq)


def sample_lambda(u, lam_min=-20.0, lam_max=20.0):
    """Map u in [0, 1] to a log-SNR with lam = -2 log tan(a u + b) (Section 2).

    b = arctan(exp(-lam_max / 2)) and a = arctan(exp(-lam_min / 2)) - b.
    """
    b = np.arctan(np.exp(-lam_max / 2))
    a = np.arctan(np.exp(-lam_min / 2)) - b
    return -2.0 * np.log(np.tan(a * np.asarray(u, dtype=float) + b))


def drop_condition(c, p_uncond, rng, null=-1):
    """Algorithm 1, line 3: replace each label by `null` with probability p_uncond.

    c is an integer array of shape (batch,). Return a new array and leave c alone.
    """
    c = np.asarray(c)
    drop = rng.random(c.shape) < p_uncond
    return np.where(drop, null, c)


def corrupt(x, lam, eps):
    """Algorithm 1, line 6: z = alpha_lam * x + sigma_lam * eps.

    x and eps have shape (batch, dim). lam is a float or an array of shape (batch,).
    """
    alpha, sigma = alpha_sigma(lam)
    shape = (-1,) + (1,) * (np.ndim(x) - 1)
    alpha, sigma = np.reshape(alpha, shape), np.reshape(sigma, shape)
    return alpha * x + sigma * eps
