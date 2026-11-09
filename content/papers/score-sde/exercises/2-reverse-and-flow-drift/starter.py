import numpy as np

BMIN, BMAX = 0.1, 20.0


def beta(t):
    """Finished for you. Appendix B, Equation 32."""
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    """Finished for you. Integral of beta from 0 to t."""
    return BMIN * t + 0.5 * (BMAX - BMIN) * t**2


def gaussian_score(x, t, m0, s0):
    """Score of p_t(x) when x(0) ~ N(m0, s0^2) and the SDE is the VP-SDE."""
    raise NotImplementedError


def reverse_sde_drift(x, t, score):
    """Drift of the reverse-time SDE, Equation 6. score = grad_x log p_t(x)."""
    raise NotImplementedError


def pf_ode_drift(x, t, score):
    """Drift of the probability-flow ODE, Equation 13. score = grad_x log p_t(x)."""
    raise NotImplementedError
