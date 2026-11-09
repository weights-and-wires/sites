import numpy as np


def adam_l2_step(theta, grad, m, v, t, lam, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam with L2 regularization (Algorithm 2, line 6 adds lam * theta to the gradient).

    Returns (theta_new, m_new, v_new) and leaves the inputs unchanged.
    """
    g = grad + lam * theta
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g * g
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    theta_new = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)
    return theta_new, m_new, v_new


def adamw_step(theta, grad, m, v, t, lam, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """AdamW (Algorithm 2, line 12 adds lam * theta to the step, outside the adaptive part).

    Returns (theta_new, m_new, v_new) and leaves the inputs unchanged.
    """
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    theta_new = theta - (alpha * m_hat / (np.sqrt(v_hat) + eps) + lam * theta)
    return theta_new, m_new, v_new
