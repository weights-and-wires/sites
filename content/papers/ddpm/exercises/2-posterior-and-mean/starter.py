import numpy as np


def make_schedule(T, beta_1=1e-4, beta_T=0.02):
    """Finished for you. Index i holds timestep t = i + 1."""
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    return betas, alphas, np.cumprod(alphas)


def q_sample(x0, t, eps, alpha_bars):
    """Finished for you. Equation 4."""
    ab = alpha_bars[np.asarray(t) - 1]
    ab = np.reshape(ab, (-1,) + (1,) * (np.ndim(x0) - 1))
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps


def posterior_mean_var(x_t, x0, t, betas, alpha_bars):
    """Return (mean, variance) of q(x_{t-1} | x_t, x_0), Equations 6 and 7.

    t is an int in 1..T. variance is a float; mean has the shape of x_t.
    """
    raise NotImplementedError


def mean_from_eps(x_t, eps_pred, t, betas, alpha_bars):
    """Return mu_theta(x_t, t) from a predicted epsilon, Equation 11."""
    raise NotImplementedError
