#!/usr/bin/env python3
"""Theorem 3: policy iteration with a compatible critic, one parameter, exact Q."""

import numpy as np
from pg_lib import (compatible_features, fit_w, linear_softmax_pi, objective,
                    q_values, state_weights, two_state_mdp, softmax_rows)

gamma, s0 = 0.5, 0
P, R = two_state_mdp()
phi = np.zeros((2, 2, 1))
phi[:, 1, 0] = 1.0
theta = np.zeros(1)
print("k, alpha_k, theta_k, J(pi_k), |dJ/dtheta|")
for k in range(1001):
    pi = linear_softmax_pi(phi, theta)
    psi = compatible_features(phi, pi)
    q = q_values(P, R, pi, gamma)
    d = state_weights(P, pi, gamma, s0)
    w = fit_w(psi, pi, d, q)
    grad = (d[:, None] * pi * psi[:, :, 0] * (psi @ w)).sum()
    alpha = 4.0 / (k + 1) ** 0.5
    if k in (0, 1, 10, 100, 1000):
        j = objective(P, R, np.stack([np.zeros(2), np.ones(2) * theta[0]], axis=1), gamma, s0)
        print(f"{k}, {alpha:.3f}, {theta[0]:.4f}, {j:.6f}, {abs(grad):.6f}")
    theta = theta + alpha * np.array([grad])

# Is the stopping point the best policy in the family? Scan theta on a fine grid.
grid = np.linspace(-8, 8, 3201)
js = [objective(P, R, np.array([[0.0, t], [0.0, t]]), gamma, s0) for t in grid]
best = int(np.argmax(js))
print(f"best J on the grid: {js[best]:.6f} at theta = {grid[best]:.2f}")
