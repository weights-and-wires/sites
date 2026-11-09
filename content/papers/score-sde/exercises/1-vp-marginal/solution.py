import numpy as np


def beta(t, bmin=0.1, bmax=20.0):
    """Finished for you. The linear schedule of Appendix B, Equation 32."""
    return bmin + t * (bmax - bmin)


def int_beta(t, bmin=0.1, bmax=20.0):
    """Integral of beta(s) from 0 to t. Works on floats and arrays."""
    return bmin * t + 0.5 * (bmax - bmin) * t**2


def vp_kernel(t, bmin=0.1, bmax=20.0):
    """Return (mean_scale, variance) of p_0t(x(t) | x(0)), Equation 33."""
    ib = int_beta(t, bmin, bmax)
    return np.exp(-0.5 * ib), 1.0 - np.exp(-ib)


def euler_maruyama_vp(x0, t_end, n_steps, rng, bmin=0.1, bmax=20.0):
    """Simulate the forward VP-SDE (Equation 11) from t = 0 to t_end.

    x0 is an array of starting values. rng is a numpy Generator.
    Returns an array shaped like x0.
    """
    x = np.array(x0, dtype=float)
    dt = t_end / n_steps
    for k in range(n_steps):
        t = k * dt
        b = beta(t, bmin, bmax)
        x = x - 0.5 * b * x * dt + np.sqrt(b * dt) * rng.standard_normal(x.shape)
    return x
