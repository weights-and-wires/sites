import numpy as np


def adam_step(theta, g, m, v, t, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """One Adam update at timestep t (t starts at 1).

    Returns (theta_new, m_new, v_new) and leaves the inputs unchanged.
    """
    raise NotImplementedError


def adam(grad_fn, theta0, steps, alpha=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Run `steps` Adam updates from theta0 and return the final parameters."""
    raise NotImplementedError
