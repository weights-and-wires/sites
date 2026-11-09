#!/usr/bin/env python3
"""Proposition 3: decoupled decay with a fixed preconditioner diag(s)^-1 equals L2 on a rescaled loss.

Run from anywhere:  python3 content/papers/adamw/scripts/fixed_preconditioner.py
"""
import numpy as np

H = np.array([1.0, 1.0, 1.0])
S = np.array([0.1, 1.0, 10.0])   # typical gradient scale of each coordinate (the paper's s)
alpha, lam, steps = 0.01, 0.01, 200
B = np.array([1.0, 1.0, 1.0])


def grad(th):
    return H * (th - B)


def run_decoupled(th):
    """theta <- (1 - lam) * theta - alpha * grad f(theta) / s (Equation 12 of Appendix A)."""
    for _ in range(steps):
        th = (1 - lam) * th - alpha * grad(th) / S
    return th


def run_l2(th, lam_vec):
    """Same optimizer without decay on f + 0.5 * sum(lam_vec * theta^2)."""
    for _ in range(steps):
        th = th - alpha * (grad(th) + lam_vec * th) / S
    return th


th0 = np.full(3, 2.0)
print("s =", S)
print("Per-step decay of theta from the L2 term, with lambda' = lambda / alpha = 1 for every coordinate:")
print("  L2 :", alpha * 1.0 / S, "   (alpha * lambda' / s)")
print("  decoupled:", np.full(3, lam), "   (lambda)")

# Equation 11: the decoupled run equals an L2 run with lambda'_i = (lam / alpha) * s_i
# (the gradient of (lambda' / 2) * ||theta * sqrt(s)||^2 is lambda' * s * theta).
a = run_decoupled(th0)
b = run_l2(th0, (lam / alpha) * S)
c = run_l2(th0, np.full(3, lam / alpha))
print("After 200 steps:")
print("  decoupled                          :", np.round(a, 6))
print("  L2 on scale-adjusted loss          :", np.round(b, 6), " max diff", f"{np.abs(a - b).max():.2e}")
print("  L2 with one lambda' = lambda/alpha :", np.round(c, 6), " max diff", f"{np.abs(a - c).max():.4f}")
