#!/usr/bin/env python3
"""Proposition 1: for SGD, decay lambda equals L2 with lambda' = lambda / alpha.

Run from anywhere:  python3 content/papers/adamw/scripts/sgd_equivalence.py
"""
import numpy as np

H = np.array([1.0, 4.0, 0.25])       # curvatures of f(theta) = 0.5 * sum(H * (theta - B)^2)
B = np.array([1.0, -2.0, 3.0])


def grad(theta):
    return H * (theta - B)


def sgd_decay(alpha, lam, steps, theta0):
    """Equation 1: theta <- (1 - lam) * theta - alpha * grad f(theta)."""
    th = theta0.copy()
    for _ in range(steps):
        th = (1 - lam) * th - alpha * grad(th)
    return th


def sgd_l2(alpha, lam_l2, steps, theta0):
    """Plain SGD on f + lam_l2 / 2 * ||theta||^2."""
    th = theta0.copy()
    for _ in range(steps):
        th = th - alpha * (grad(th) + lam_l2 * th)
    return th


def sgdm_l2(alpha, lam_l2, beta, steps, theta0):
    """Algorithm 1 with L2: line 6 puts lam * theta in the gradient, then momentum."""
    th, m = theta0.copy(), np.zeros_like(theta0)
    for _ in range(steps):
        g = grad(th) + lam_l2 * th
        m = beta * m + alpha * g
        th = th - m
    return th


def sgdw(alpha, lam, beta, steps, theta0):
    """Algorithm 1 with decoupled decay (SGDW): line 9 subtracts lam * theta."""
    th, m = theta0.copy(), np.zeros_like(theta0)
    for _ in range(steps):
        m = beta * m + alpha * grad(th)
        th = th - m - lam * th
    return th


theta0 = np.array([5.0, 5.0, 5.0])
alpha, lam, steps = 0.1, 0.01, 50

a = sgd_decay(alpha, lam, steps, theta0)
b = sgd_l2(alpha, lam / alpha, steps, theta0)
c = sgd_l2(alpha, lam, steps, theta0)
print("Proposition 1, alpha=0.1, lambda=0.01, 50 steps")
print("  decay             :", np.round(a, 6))
print("  L2, lambda'=lam/a :", np.round(b, 6), " max diff", f"{np.abs(a - b).max():.2e}")
print("  L2, lambda'=lam   :", np.round(c, 6), " max diff", f"{np.abs(a - c).max():.2e}")

print("Coupling: keep lambda' = 0.1 fixed and change alpha")
for al in (0.1, 0.05, 0.025):
    print(f"  alpha={al}: decay per step on theta = alpha * lambda' = {al * 0.1:.5f}")
print("Decoupled: lambda = 0.01 gives decay per step 0.01000 at every alpha")

print("With momentum beta=0.9 the two forms stop agreeing (Algorithm 1, lines 6 to 9)")
d = sgdm_l2(alpha, lam / alpha, 0.9, steps, theta0)
e = sgdw(alpha, lam, 0.9, steps, theta0)
print("  SGD+L2 :", np.round(d, 6))
print("  SGDW   :", np.round(e, 6), " max diff", f"{np.abs(d - e).max():.4f}")
