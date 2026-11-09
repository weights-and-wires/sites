import numpy as np


def make_schedule(T, beta_1=1e-4, beta_T=0.02):
    """Return (betas, alphas, alpha_bars), each an array of length T.

    betas increase linearly from beta_1 to beta_T. Index i holds timestep t = i + 1.
    """
    betas = np.linspace(beta_1, beta_T, T)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def q_sample(x0, t, eps, alpha_bars):
    """Sample x_t ~ q(x_t | x_0) with the closed form (Equation 4).

    x0 and eps have the same shape (batch, dim). t is an int or an int array of
    shape (batch,), with values in 1..T. alpha_bars comes from make_schedule.
    """
    ab = alpha_bars[np.asarray(t) - 1]
    ab = np.reshape(ab, (-1,) + (1,) * (np.ndim(x0) - 1))
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps
