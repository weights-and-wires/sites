import numpy as np

# Appendix D.2, Equation 10.
E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def frontier(c):
    """Return (n_opt, d_opt) that minimise E + A/N**ALPHA + B/D**BETA subject to 6*N*D = c."""
    g = (ALPHA * A / (BETA * B)) ** (1 / (ALPHA + BETA))
    a = BETA / (ALPHA + BETA)
    b = ALPHA / (ALPHA + BETA)
    return g * (c / 6) ** a, (c / 6) ** b / g
