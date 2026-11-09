import numpy as np


def adam_step(theta, g, m, v, t, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam update at timestep t (t starts at 1).

    Returns (theta_new, m_new, v_new) and leaves the inputs unchanged.
    """
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g * g
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    theta_new = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)
    return theta_new, m_new, v_new


def adam(grad_fn, theta0, steps, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Run `steps` Adam updates from theta0 and return the final parameters."""
    theta = np.array(theta0, dtype=float)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        theta, m, v = adam_step(theta, grad_fn(theta), m, v, t, alpha, beta1, beta2, eps)
    return theta
