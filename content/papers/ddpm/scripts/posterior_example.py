"""Compute the numbers for the forward process posterior examples.

Run: python3 scripts/posterior_example.py
"""

import numpy as np

betas = np.array([0.1, 0.2, 0.3, 0.4])
alphas = 1 - betas
alpha_bars = np.cumprod(alphas)

t = 3
x0, xt = 2.0, 0.5
ab_t, ab_prev = alpha_bars[t - 1], alpha_bars[t - 2]
beta_t, alpha_t = betas[t - 1], alphas[t - 1]

# Equations 6 and 7.
c0 = np.sqrt(ab_prev) * beta_t / (1 - ab_t)
ct = np.sqrt(alpha_t) * (1 - ab_prev) / (1 - ab_t)
mean = c0 * x0 + ct * xt
beta_tilde = (1 - ab_prev) / (1 - ab_t) * beta_t
print(f"t={t}  alpha_bar_t={ab_t:.4f}  alpha_bar_(t-1)={ab_prev:.4f}")
print(f"coefficient on x0 = {c0:.4f}, on x_t = {ct:.4f}")
print(f"posterior mean = {mean:.4f}, beta_tilde = {beta_tilde:.4f}, beta_t = {beta_t:.4f}")

# Bayes' rule check: prior q(x_{t-1}|x0) times likelihood q(x_t|x_{t-1}).
prec = 1 / (1 - ab_prev) + alpha_t / beta_t
mean_bayes = (np.sqrt(ab_prev) * x0 / (1 - ab_prev) + np.sqrt(alpha_t) * xt / beta_t) / prec
print(f"Bayes: mean = {mean_bayes:.4f}, variance = {1 / prec:.4f}")

# Equation 10/11: the same mean written with epsilon.
eps = (xt - np.sqrt(ab_t) * x0) / np.sqrt(1 - ab_t)
mean_eps = (xt - beta_t / np.sqrt(1 - ab_t) * eps) / np.sqrt(alpha_t)
print(f"epsilon implied by (x0, x_t) = {eps:.4f}")
print(f"mean from epsilon form = {mean_eps:.4f}")

# t = 1: alpha_bar_0 = 1, so beta_tilde_1 = 0 and the posterior mean is x0.
print("beta_tilde_1 =", (1 - 1.0) / (1 - alpha_bars[0]) * betas[0])
