import numpy as np

# Appendix K, Table 2. GAMMA_POST is the row labeled "gamma" (read as gamma_post).
B, BETA = 1.806e4, 0.4965
C_T, GAMMA_D, GAMMA_N, GAMMA_POST = 0.0598, 0.5068, 0.3439, 0.5907


def delta_ptq(n, d, p_post):
    """Return C_T * d**GAMMA_D / n**GAMMA_N * exp(-p_post / GAMMA_POST) (Equation 2)."""
    raise NotImplementedError


def d_crit(n, p_post):
    """Return the token count at which more data stops helping after PTQ.

    It solves B * BETA * D**(-BETA - 1) = d(delta_ptq)/dD (Equations 12 and 13).
    """
    raise NotImplementedError
