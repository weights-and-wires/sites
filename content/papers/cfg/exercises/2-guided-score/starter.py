import numpy as np


def guided_eps(eps_c, eps_u, w):
    """Equation 6: (1 + w) * eps(z, c) - w * eps(z).

    eps_c and eps_u are arrays of the same shape; w is a float.
    """
    raise NotImplementedError


def implicit_classifier_grad(eps_c, eps_u, sigma):
    """Gradient of the implicit classifier's log probability (Section 3.2):

    grad_z log p_i(c | z) = -(1 / sigma) * (eps(z, c) - eps(z)).
    """
    raise NotImplementedError


def classifier_guided_eps(eps_c, grad_log_pc, sigma, w):
    """Classifier guidance (Section 3.1): eps(z, c) - w * sigma * grad_z log p(c | z)."""
    raise NotImplementedError
