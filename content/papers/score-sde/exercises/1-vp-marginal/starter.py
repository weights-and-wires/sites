import numpy as np


def beta(t, bmin=0.1, bmax=20.0):
    """Finished for you. The linear schedule of Appendix B, Equation 32."""
    return bmin + t * (bmax - bmin)


def int_beta(t, bmin=0.1, bmax=20.0):
    """Integral of beta(s) from 0 to t. Works on floats and arrays."""
    raise NotImplementedError


def vp_kernel(t, bmin=0.1, bmax=20.0):
    """Return (mean_scale, variance) of p_0t(x(t) | x(0)), Equation 33."""
    raise NotImplementedError


def euler_maruyama_vp(x0, t_end, n_steps, rng, bmin=0.1, bmax=20.0):
    """Simulate the forward VP-SDE (Equation 11) from t = 0 to t_end.

    x0 is an array of starting values. rng is a numpy Generator.
    Returns an array shaped like x0.
    """
    raise NotImplementedError
