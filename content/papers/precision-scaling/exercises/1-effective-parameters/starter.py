import numpy as np

# Appendix K, Table 2.
A, B, E, ALPHA = 4.299e3, 1.806e4, 2.7648, 0.4965
GAMMA = {"w": 2.6745, "a": 2.2102, "kv": 0.9578}
OFFSET = {"w": 0.3037, "a": 1.4072, "kv": 2.4185}


def eff_factor(p, gamma, offset=0.0):
    """Return 1 - exp(-p / gamma + offset), one factor of Equation 4."""
    raise NotImplementedError


def n_eff(n, p_w, p_a, p_kv, offsets=False):
    """Return N_eff of Equation 4: n times the product of the three factors.

    With offsets=True, use the Table 2 offsets OFFSET; otherwise use 0.
    """
    raise NotImplementedError


def loss_hat(n, d, p_w, p_a, p_kv, offsets=False):
    """Return A * N_eff**-ALPHA + B * d**-ALPHA + E (Section 5, without PTQ)."""
    raise NotImplementedError
