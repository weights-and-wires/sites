import numpy as np


def adam_l2_step(theta, grad, m, v, t, lam, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam with L2 regularization (Algorithm 2, line 6 adds lam * theta to the gradient).

    Returns (theta_new, m_new, v_new) and leaves the inputs unchanged.
    """
    raise NotImplementedError


def adamw_step(theta, grad, m, v, t, lam, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """AdamW (Algorithm 2, line 12 adds lam * theta to the step, outside the adaptive part).

    Returns (theta_new, m_new, v_new) and leaves the inputs unchanged.
    """
    raise NotImplementedError
