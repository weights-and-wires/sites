"""Hand-checkable drifts for the VP-SDE with Gaussian data (Equations 6 and 13).

Data x(0) ~ N(2, 0.5^2). Then p_t is Gaussian with mean 2 m(t) and variance
m(t)^2 * 0.25 + (1 - m(t)^2), where m(t) = exp(-0.5 int beta). Prints the score
and the three drifts at t = 0.5, x = 1.

Run: python3 scripts/drift_example.py
"""

import numpy as np

BMIN, BMAX = 0.1, 20.0
M0, S0 = 2.0, 0.5


def beta(t):
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


t, x = 0.5, 1.0
m = np.exp(-0.5 * int_beta(t))
mean_t = m * M0
var_t = m**2 * S0**2 + (1 - m**2)
score = -(x - mean_t) / var_t
f = -0.5 * beta(t) * x
g2 = beta(t)

print(f"beta(t)                 = {beta(t):.4f}")
print(f"m(t)                    = {m:.4f}")
print(f"mean of p_t             = {mean_t:.4f}")
print(f"variance of p_t         = {var_t:.4f}")
print(f"score at x = {x}         = {score:.4f}")
print(f"forward drift f         = {f:.4f}")
print(f"g^2                     = {g2:.4f}")
print(f"reverse-SDE drift       = {f - g2 * score:.4f}   (f - g^2 * score, Equation 6)")
print(f"probability-flow drift  = {f - 0.5 * g2 * score:.4f}   (f - g^2 * score / 2, Equation 13)")
