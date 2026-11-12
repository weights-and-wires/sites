import numpy as np

# Appendix K, Table 2.
A, ALPHA = 4.299e3, 0.4965


def n_eff_fraction(loss, loss_inf, n):
    """Return f = N_eff / N from a measured loss (Appendix M, Equation 30).

    loss is the loss at some weight precision, loss_inf is the loss of the
    same (N, D) run at full precision, and n is the parameter count.
    """
    loss = np.asarray(loss, dtype=float)
    return ((loss - loss_inf) / A + n**-ALPHA) ** (-1.0 / ALPHA) / n


def fit_gamma(ps, fracs):
    """Fit f(P) = 1 - exp(-P / gamma) to the points (ps, fracs).

    Use ln(1 - f) = -P / gamma and a least-squares line through the origin.
    Return gamma.
    """
    p = np.asarray(ps, dtype=float)
    y = np.log(1.0 - np.asarray(fracs, dtype=float))
    return float(-np.sum(p * p) / np.sum(p * y))
