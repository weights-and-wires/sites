"""Compute the numbers for the softmax Fisher matrix and the damped natural gradient.

One state, three actions, a tabular softmax policy pi(a) = exp(theta_a) / sum_b exp(theta_b),
so psi(a) = grad_theta log pi(a) = e_a - pi. The action values Q are held fixed.

Checks three things from Kakade (2001):
  - F = diag(pi) - pi pi^T (Equation 2) is singular, so you damp it before solving.
  - The natural gradient w is Q up to a constant shift (Theorem 1 with a tabular policy),
    so f(a) = w . psi(a) equals the advantage Q(a) - V.
  - A small natural gradient step multiplies each probability by 1 + alpha * (Q - V)
    (Theorem 3), up to O(alpha^2).

Run: python3 softmax_fisher.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

pi = np.array([0.5, 0.3, 0.2])
Q = np.array([1.0, 0.0, 0.5])
V = pi @ Q
A = Q - V
print(f"pi                         = {pi}")
print(f"Q                          = {Q}")
print(f"V = pi . Q                 = {V:.4f}")
print(f"A = Q - V                  = {A}")

# Equation 2 by definition: F = E_pi[psi psi^T], with psi(a) = e_a - pi.
psi = np.eye(3) - pi  # row a is psi(a)
F = sum(pi[a] * np.outer(psi[a], psi[a]) for a in range(3))
print("\nF (Equation 2) =")
print(F)
print(f"diag(pi) - pi pi^T matches = {np.allclose(F, np.diag(pi) - np.outer(pi, pi))}")
print(f"F @ [1, 1, 1]              = {F @ np.ones(3)}")
print(f"eigenvalues of F           = {np.linalg.eigvalsh(F)}")

# Equation 1 for one state (rho = 1): grad = sum_a grad pi(a) Q(a) = sum_a pi(a) psi(a) Q(a).
g = sum(pi[a] * psi[a] * Q[a] for a in range(3))
print(f"\ngrad eta = pi * (Q - V)    = {g}")
print(f"entries sum to             = {g.sum():.4f}")

print("\nDamped natural gradient (F + lambda I)^-1 grad:")
for lam in (1.0, 0.1, 1e-3, 1e-6):
    w = np.linalg.solve(F + lam * np.eye(3), g)
    print(f"  lambda = {lam:<7g} -> {w}")
print(f"pinv(F) @ grad             = {np.linalg.pinv(F) @ g}")
w0 = np.linalg.pinv(F) @ g
print(f"Q - mean(Q)                = {Q - Q.mean()}")
print(f"w - A                      = {w0 - A}  (a multiple of [1, 1, 1], which F ignores)")
print(f"f(a) = w . psi(a)          = {psi @ w0}")
print(f"A                          = {A}")

# Theorem 3: pi' = pi * (1 + alpha * f(a)) + O(alpha^2), where f(a) = w . psi(a) = A(a) here.
print("\nTheorem 3: natural gradient step theta' = theta + alpha * w")
theta = np.log(pi)
for alpha in (0.1, 0.01):
    t2 = theta + alpha * w0
    pi2 = np.exp(t2) / np.exp(t2).sum()
    first_order = pi * (1 + alpha * A)
    print(f"  alpha = {alpha}")
    print(f"    exact pi'              = {pi2}")
    print(f"    pi * (1 + alpha * A)   = {first_order}")
    print(f"    max abs error          = {np.abs(pi2 - first_order).max():.2e}")
