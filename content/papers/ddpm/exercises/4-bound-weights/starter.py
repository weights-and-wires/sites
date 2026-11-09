import numpy as np


def make_schedule(T, beta_1=1e-4, beta_T=0.02):
    """Finished for you. Index i holds timestep t = i + 1."""
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    return betas, alphas, np.cumprod(alphas)


def bound_weight(t, betas, alpha_bars, sigma2):
    """Equation 12: beta_t^2 / (2 sigma_t^2 alpha_t (1 - alpha_bar_t)). t is an int in 1..T."""
    raise NotImplementedError


def kl_term_from_eps(eps, eps_pred, t, betas, alpha_bars, sigma2):
    """w_t * ||eps - eps_pred||^2, summed over the data dimensions (Equation 12 without the expectation)."""
    raise NotImplementedError
