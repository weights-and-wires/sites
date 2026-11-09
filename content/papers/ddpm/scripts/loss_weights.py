"""Compute the numbers for the loss-weighting example (Equations 12 and 14).

Run: python3 scripts/loss_weights.py
"""

import numpy as np

T = 1000
betas = np.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)
alpha_bars_prev = np.concatenate([[1.0], alpha_bars[:-1]])
beta_tilde = (1 - alpha_bars_prev) / (1 - alpha_bars) * betas

# Equation 12 weight on ||eps - eps_theta||^2 for the two variance choices.
w_beta = betas**2 / (2 * betas * alphas * (1 - alpha_bars))
with np.errstate(divide="ignore"):
    w_tilde = np.where(beta_tilde > 0, betas**2 / (2 * beta_tilde * alphas * (1 - alpha_bars)), np.nan)

print("sigma^2 = beta_t:        w_1 = %.4f  w_10 = %.4f  w_100 = %.4f  w_1000 = %.4f" % (w_beta[0], w_beta[9], w_beta[99], w_beta[999]))
print("sigma^2 = beta_tilde_t:  w_2 = %.4f  w_10 = %.4f  w_100 = %.4f  w_1000 = %.4f" % (w_tilde[1], w_tilde[9], w_tilde[99], w_tilde[999]))
print("ratio w_1 / w_1000 (sigma^2 = beta_t) = %.1f" % (w_beta[0] / w_beta[999]))
print("L_simple sets every weight to 1, so relative to the bound it scales term t by 1/w_t:")
print("  t=1: x%.2f   t=100: x%.2f   t=1000: x%.2f" % (1 / w_beta[0], 1 / w_beta[99], 1 / w_beta[999]))
