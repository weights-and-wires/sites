import numpy as np

# Table 2 of the paper: the fit of Equation 1.5
ALPHA_N, ALPHA_D, N_C, D_C = 0.076, 0.103, 6.4e13, 1.8e13


def loss_nd(n, d, alpha_n=ALPHA_N, alpha_d=ALPHA_D, n_c=N_C, d_c=D_C):
    """L(N, D) = [(N_c / N) ** (alpha_N / alpha_D) + D_c / D] ** alpha_D."""
    raise NotImplementedError


def loss_n_only(n, alpha_n=ALPHA_N, n_c=N_C):
    """L(N) = (N_c / N) ** alpha_N, the loss with unlimited data."""
    raise NotImplementedError


def overfit_penalty(n, d):
    """delta L = L(N, D) / L(N, infinity) - 1."""
    raise NotImplementedError


def min_tokens(n):
    """Dataset size that keeps overfitting within seed noise: D >= 5e3 * N ** 0.74."""
    raise NotImplementedError
