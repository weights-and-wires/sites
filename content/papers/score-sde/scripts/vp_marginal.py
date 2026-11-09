"""VP-SDE marginals (Section 3.4, Equations 11, 32 and 33; Appendix B, Equation 26).

Prints the closed-form mean scale and variance of p_0t(x(t) | x(0)) for the
paper's linear beta(t), compares them with the discrete DDPM schedule of
Section 2.2, and checks that unit-variance data keeps unit variance.

Run: python3 scripts/vp_marginal.py
"""

import numpy as np

BMIN, BMAX = 0.1, 20.0  # beta-bar_min and beta-bar_max, Appendix C


def beta(t):
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


def mean_scale(t):
    return np.exp(-0.5 * int_beta(t))


def var(t):
    return 1.0 - np.exp(-int_beta(t))


print("t      beta(t)   int beta   mean scale   variance   std")
for t in (0.001, 0.1, 0.25, 0.5, 1.0):
    print(f"{t:5.3f}  {beta(t):7.3f}  {int_beta(t):8.4f}  {mean_scale(t):10.6f}  {var(t):9.6f}  {np.sqrt(var(t)):.4f}")

# Discrete DDPM chain from the same beta(t): beta_i = beta(i/N) / N (Appendix B, Equation 23).
N = 1000
i = np.arange(1, N + 1)
betas_disc = beta(i / N) / N
abar = np.cumprod(1 - betas_disc)
print()
print(f"Discrete chain with N = {N}: alpha_bar_N = {abar[-1]:.3e}")
print(f"Continuous exp(-int_0^1 beta)   = {np.exp(-int_beta(1.0)):.3e}")
print(f"Ho et al. linspace(1e-4, 0.02)  = {np.cumprod(1 - np.linspace(1e-4, 0.02, N))[-1]:.3e}")
print(f"beta_1 = {betas_disc[0]:.6f}, beta_N = {betas_disc[-1]:.6f}")

# Variance preservation (Appendix B, Equation 26): Sigma(t) = 1 + exp(-int beta) (Sigma(0) - 1).
print()
print("Variance of x(t) for data variance s0^2 (Equation 26):")
for s0sq in (0.25, 1.0, 4.0):
    row = [1.0 + np.exp(-int_beta(t)) * (s0sq - 1.0) for t in (0.0, 0.1, 0.5, 1.0)]
    print(f"  s0^2 = {s0sq:4.2f}: " + "  ".join(f"{v:.4f}" for v in row) + "   (t = 0, 0.1, 0.5, 1)")
