import numpy as np


def l2_precond_step(theta, grad, s, alpha, lam_l2):
    """One step with preconditioner diag(s)^-1 on the loss plus an L2 term.

    The L2 term is (lam_l2 / 2) * theta**2 per coordinate, so it adds lam_l2 * theta to the gradient.
    """
    return theta - alpha * (grad + lam_l2 * theta) / s


def decoupled_precond_step(theta, grad, s, alpha, lam):
    """One step with preconditioner diag(s)^-1 and decoupled weight decay lam."""
    return (1 - lam) * theta - alpha * grad / s


def equivalent_l2(lam, alpha, s):
    """Per-coordinate L2 coefficients that make l2_precond_step match decoupled decay lam."""
    return lam / alpha * np.asarray(s, dtype=float)
