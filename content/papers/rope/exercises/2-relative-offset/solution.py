import numpy as np


def rotary_matrix(m, d, base=10000.0):
    """Return the (d, d) block-diagonal rotary matrix R_{Theta, m} (Equation 15)."""
    R = np.zeros((d, d))
    for j in range(d // 2):
        theta = base ** (-2.0 * j / d)
        c, s = np.cos(m * theta), np.sin(m * theta)
        R[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = [[c, -s], [s, c]]
    return R


def rope_score(q, k, m, n, base=10000.0):
    """Return q_m . k_n as q^T R_{n-m} k (Equation 16). q and k are unrotated."""
    d = len(q)
    return float(q @ rotary_matrix(n - m, d, base) @ k)
