"""Compute the numbers for the metric and reparameterisation examples.

Part A: a one-parameter policy and a rescaled copy of it. The plain gradient step
depends on the units of the parameter. The natural gradient step doesn't.

Part B: two states with different visit frequencies, as in the paper's Figure 1B
discussion (Section 5). The plain gradient is weighted by the state distribution.
The natural gradient isn't.

Notation follows Kakade (2001): rho is the state distribution, F the Fisher matrix
(Equations 2 and 3), and the natural gradient is F^-1 grad (Equation 4). Q is held
fixed at the values shown, so this is a snapshot of one gradient, not a full MDP.

Run: python3 invariance.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def kl_bernoulli(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


print("== Part A: one state, two actions, pi(1) = sigmoid(theta), Q = [0, 1]")
theta = 0.0
p = sigmoid(theta)
grad = p * (1 - p)  # d eta / d theta, with eta = pi(1) * 1
fisher = p * (1 - p)  # E[(d log pi / d theta)^2]
print(f"pi(1) at theta = 0        = {p:.4f}")
print(f"grad eta                  = {grad:.4f}")
print(f"F                         = {fisher:.4f}")
print(f"natural gradient F^-1 grad = {grad / fisher:.4f}")

alpha = 0.1
for c in (1.0, 10.0):
    # Reparameterise theta = c * u. Then grad_u = c * grad_theta and F_u = c^2 * F_theta.
    grad_u = c * grad
    fisher_u = c**2 * fisher
    du_plain = alpha * grad_u
    du_nat = alpha * grad_u / fisher_u
    dtheta_plain = c * du_plain
    dtheta_nat = c * du_nat
    p_plain = sigmoid(theta + dtheta_plain)
    p_nat = sigmoid(theta + dtheta_nat)
    print(f"-- scale c = {c:g}, step size alpha = {alpha}")
    print(f"   grad_u = {grad_u:.4f}, F_u = {fisher_u:.4f}")
    print(f"   plain:   delta u = {du_plain:.4f}, delta theta = {dtheta_plain:.4f}, pi(1) = {p_plain:.4f}, KL(old||new) = {kl_bernoulli(p, p_plain):.5f}")
    print(f"   natural: delta u = {du_nat:.4f}, delta theta = {dtheta_nat:.4f}, pi(1) = {p_nat:.4f}, KL(old||new) = {kl_bernoulli(p, p_nat):.5f}")

# Damping adds lambda to the diagonal of F, and that term is not invariant.
lam = 0.1
print(f"-- damped natural step, lambda = {lam}, alpha = {alpha}")
for c in (1.0, 10.0):
    grad_u = c * grad
    fisher_u = c**2 * fisher
    du = alpha * grad_u / (fisher_u + lam)
    print(f"   c = {c:g}: delta theta = {c * du:.4f}")

print()
print("== Part B: two states, pi(1; s) = sigmoid(theta_s), rho = [0.8, 0.2]")
rho = np.array([0.8, 0.2])
theta = np.array([np.log(0.8 / 0.2), np.log(0.1 / 0.9)])  # pi(1; s) = [0.8, 0.1]
pi1 = sigmoid(theta)
dQ = np.array([0.5, 1.0])  # Q(s, 1) - Q(s, 0), held fixed
grad = rho * pi1 * (1 - pi1) * dQ  # Equation 1
F = np.diag(rho * pi1 * (1 - pi1))  # Equation 3: sum_s rho(s) F_s
print(f"pi(1; s)                  = {pi1}")
print(f"rho                       = {rho}")
print(f"grad eta (Equation 1)     = {grad}")
print(f"ratio grad[0] / grad[1]   = {grad[0] / grad[1]:.3f}")
print(f"F diagonal                = {np.diag(F)}")
nat = np.linalg.solve(F, grad)
print(f"natural gradient          = {nat}")
print(f"ratio nat[0] / nat[1]     = {nat[0] / nat[1]:.3f}")
