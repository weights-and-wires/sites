#!/usr/bin/env python3
"""Numbers for Theorem 1 (Equation 2) on the 2-state MDP, plus a random-MDP check."""

import numpy as np
from pg_lib import (finite_difference_gradient, objective, policy_gradient,
                    q_values, random_mdp, softmax_rows, state_values,
                    state_weights, two_state_mdp)

np.set_printoptions(precision=5, suppress=True)
gamma, s0 = 0.5, 0
P, R = two_state_mdp()
theta = np.zeros((2, 2))
pi = softmax_rows(theta)
V = state_values(P, R, pi, gamma)
Q = q_values(P, R, pi, gamma)
d = state_weights(P, pi, gamma, s0)
print("pi =", pi.tolist(), "gamma =", gamma)
print("V =", V, "  J = V(s0) =", objective(P, R, theta, gamma, s0))
print("Q =", Q.tolist())
print("d =", d, " sum(d) =", d.sum(), " 1/(1-gamma) =", 1 / (1 - gamma))
g = policy_gradient(P, R, theta, gamma, s0)
fd = finite_difference_gradient(P, R, theta, gamma, s0)
print("Theorem 1 gradient wrt theta[s, a]:", g.tolist())
print("finite differences              :", fd.tolist())
print("max abs difference:", float(np.abs(g - fd).max()))

print()
print("Naive gradient that drops the state weights d (uses d = 1 for every state):")
naive = pi * (Q - (pi * Q).sum(axis=1, keepdims=True))
print(naive.tolist())

print()
P3, R3 = random_mdp(3, 2, seed=0)
th = np.random.default_rng(1).normal(size=(3, 2))
g3 = policy_gradient(P3, R3, th, 0.9, 0)
fd3 = finite_difference_gradient(P3, R3, th, 0.9, 0)
print("random 3-state MDP, gamma = 0.9: max abs difference", float(np.abs(g3 - fd3).max()))
