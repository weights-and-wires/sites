import numpy as np


def sgd_l2_step(theta, grad, m, alpha, lam, beta1):
    """SGD with momentum on the L2-regularized gradient (Algorithm 1, lines 6, 8, 9).

    Returns (theta_new, m_new) and leaves the inputs unchanged.
    """
    raise NotImplementedError


def sgdw_step(theta, grad, m, alpha, lam, beta1):
    """SGD with momentum and decoupled weight decay (Algorithm 1, lines 8 and 9).

    Returns (theta_new, m_new) and leaves the inputs unchanged.
    """
    raise NotImplementedError


def l2_factor(lam, alpha):
    """The L2 factor lambda' that matches weight decay lam for plain SGD (Proposition 1)."""
    raise NotImplementedError
