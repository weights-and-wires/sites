import numpy as np

# Table 2 of the paper: the fit of Equation 1.5
ALPHA_N, ALPHA_D, N_C, D_C = 0.076, 0.103, 6.4e13, 1.8e13


def loss_nd(n, d, alpha_n=ALPHA_N, alpha_d=ALPHA_D, n_c=N_C, d_c=D_C):
    """L(N, D) = [(N_c / N) ** (alpha_N / alpha_D) + D_c / D] ** alpha_D."""
    n = np.asarray(n, dtype=float)
    d = np.asarray(d, dtype=float)
    return ((n_c / n) ** (alpha_n / alpha_d) + d_c / d) ** alpha_d


def loss_n_only(n, alpha_n=ALPHA_N, n_c=N_C):
    """L(N) = (N_c / N) ** alpha_N, the loss with unlimited data."""
    return (n_c / np.asarray(n, dtype=float)) ** alpha_n


def overfit_penalty(n, d):
    """delta L = L(N, D) / L(N, infinity) - 1."""
    return loss_nd(n, d) / loss_n_only(n) - 1


def min_tokens(n):
    """Dataset size that keeps overfitting within seed noise: D >= 5e3 * N ** 0.74."""
    return 5e3 * np.asarray(n, dtype=float) ** 0.74
