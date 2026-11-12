#!/usr/bin/env python3
"""Numbers for Theorem 2 (Equations 3 to 5) on the 2-state MDP with one shared feature."""

import numpy as np
from pg_lib import (compatible_features, fit_w, policy_gradient, q_values,
                    random_mdp, softmax_rows, state_weights, two_state_mdp,
                    linear_softmax_pi)

np.set_printoptions(precision=5, suppress=True)
gamma, s0 = 0.5, 0
P, R = two_state_mdp()
# One parameter: phi(s, a) = 1 for a = 1 and 0 for a = 0, in both states.
phi = np.zeros((2, 2, 1))
phi[:, 1, 0] = 1.0
theta = np.zeros(1)
pi = linear_softmax_pi(phi, theta)
psi = compatible_features(phi, pi)
q = q_values(P, R, pi, gamma)
d = state_weights(P, pi, gamma, s0)
print("psi(s, a) =", psi[:, :, 0].tolist())
w = fit_w(psi, pi, d, q)
f = psi @ w
print("w =", w)
print("f_w(s, a) =", f.tolist(), "  Q(s, a) =", q.tolist())
print("zero mean per state, sum_a pi f_w =", (pi * f).sum(axis=1))
true_grad = (d[:, None] * pi * psi[:, :, 0] * q).sum()
fw_grad = (d[:, None] * pi * psi[:, :, 0] * f).sum()
tab = policy_gradient(P, R, np.zeros((2, 2)), gamma, s0)
print("true gradient d J / d theta           :", true_grad)
print("gradient with f_w in place of Q (Eq 5):", fw_grad)
print("tabular gradient mapped to theta (g[:, 1].sum()):", tab[:, 1].sum())
print("residual orthogonality (Eq 3), sum d pi psi (Q - f_w):", (d[:, None] * pi * psi[:, :, 0] * (q - f)).sum())

print()
# A model with more state-action pairs than parameters: Q is not representable.
P3, R3 = random_mdp(3, 3, seed=2)
phi3 = np.random.default_rng(4).normal(size=(3, 3, 2))
th3 = np.array([0.3, -0.2])
pi3 = linear_softmax_pi(phi3, th3)
psi3 = compatible_features(phi3, pi3)
q3 = q_values(P3, R3, pi3, 0.9)
d3 = state_weights(P3, pi3, 0.9, 0)
w3 = fit_w(psi3, pi3, d3, q3)
f3 = psi3 @ w3
gt = np.einsum("sa,sak,sa->k", d3[:, None] * pi3, psi3, q3)
gf = np.einsum("sa,sak,sa->k", d3[:, None] * pi3, psi3, f3)
print("3 states x 3 actions, 2 parameters")
print("  max |Q - f_w|:", float(np.abs(q3 - f3).max()), "(f_w is not Q)")
print("  true gradient:", gt, " with f_w:", gf)
