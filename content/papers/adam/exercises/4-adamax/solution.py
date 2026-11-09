import numpy as np


def adamax_step(theta, g, m, u, t, alpha=2e-3, beta1=0.9, beta2=0.999):
    """One AdaMax update at timestep t (t starts at 1).

    Returns (theta_new, m_new, u_new) and leaves the inputs unchanged.
    """
    m_new = beta1 * m + (1 - beta1) * g
    u_new = np.maximum(beta2 * u, np.abs(g))
    theta_new = theta - (alpha / (1 - beta1**t)) * m_new / u_new
    return theta_new, m_new, u_new


def adamax(grad_fn, theta0, steps, alpha=2e-3, beta1=0.9, beta2=0.999):
    """Run `steps` AdaMax updates from theta0 and return the final parameters."""
    theta = np.array(theta0, dtype=float)
    m = np.zeros_like(theta)
    u = np.zeros_like(theta)
    for t in range(1, steps + 1):
        theta, m, u = adamax_step(theta, grad_fn(theta), m, u, t, alpha, beta1, beta2)
    return theta
