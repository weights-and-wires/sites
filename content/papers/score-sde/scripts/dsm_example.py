"""Denoising score matching target and weighting for the VP-SDE (Equation 7).

Prints, for one (x0, t, eps), the target score grad log p_0t(x(t) | x(0)), the
weight lambda(t) = 1 / E||target||^2 = sigma^2(t), and the loss of three
candidate score models. Also shows the equivalence with epsilon prediction.

Run: python3 scripts/dsm_example.py
"""

import numpy as np

BMIN, BMAX = 0.1, 20.0


def int_beta(t):
    return BMIN * t + 0.5 * t**2 * (BMAX - BMIN)


t, x0, eps = 0.5, 2.0, -1.0
m = np.exp(-0.5 * int_beta(t))
sig2 = 1 - np.exp(-int_beta(t))
sig = np.sqrt(sig2)
xt = m * x0 + sig * eps
target = -(xt - m * x0) / sig2

print(f"m(t) = {m:.4f}, sigma^2(t) = {sig2:.4f}, sigma(t) = {sig:.4f}")
print(f"x(t) = m x0 + sigma eps = {xt:.4f}")
print(f"target score = -(x(t) - m x0) / sigma^2 = {target:.4f}")
print(f"-eps / sigma                            = {-eps / sig:.4f}")
print(f"E[target^2] = 1/sigma^2 = {1 / sig2:.4f}, so lambda = sigma^2 = {sig2:.4f}")
print()
for name, s in (("exact (-eps/sigma)", target), ("zero score", 0.0), ("score = -x(t)", -xt)):
    unweighted = (s - target) ** 2
    weighted = sig2 * unweighted
    eps_hat = -sig * s
    print(f"{name:20s} unweighted {unweighted:8.4f}   weighted {weighted:8.4f}   (eps_hat - eps)^2 = {(eps_hat - eps) ** 2:8.4f}")
