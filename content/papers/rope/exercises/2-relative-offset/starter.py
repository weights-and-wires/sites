import numpy as np


def rotary_matrix(m, d, base=10000.0):
    """Return the (d, d) block-diagonal rotary matrix R_{Theta, m} (Equation 15)."""
    raise NotImplementedError


def rope_score(q, k, m, n, base=10000.0):
    """Return q_m . k_n as q^T R_{n-m} k (Equation 16). q and k are unrotated."""
    raise NotImplementedError
