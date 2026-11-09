import numpy as np


def adamax_step(theta, g, m, u, t, alpha=2e-3, beta1=0.9, beta2=0.999):
    """One AdaMax update at timestep t (t starts at 1).

    Returns (theta_new, m_new, u_new) and leaves the inputs unchanged.
    """
    raise NotImplementedError


def adamax(grad_fn, theta0, steps, alpha=2e-3, beta1=0.9, beta2=0.999):
    """Run `steps` AdaMax updates from theta0 and return the final parameters."""
    raise NotImplementedError
