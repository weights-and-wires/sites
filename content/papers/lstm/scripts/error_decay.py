"""Compute the numbers for the error-backflow examples (Section 3.1 and 3.2).

Part 1 evaluates the one-unit chain behind Equation 3.2 and the bound in
Section 3.1.5. Part 2 measures ||d h_t / d h_0|| for a tanh RNN and for the
constant error carousel (CEC) of Section 3.2.2.

Run: python3 scripts/error_decay.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=False)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Part 1: one unit with a self-connection (Equation 3.2 with n = 1).
# Each step scales the error by f'(net) * w. For the logistic sigmoid, f' <= 0.25.
fprime_max = sigmoid(0.0) * (1 - sigmoid(0.0))
print(f"max of the logistic derivative f' = {fprime_max}")
for w in (1.0, 3.0, 4.0):
    factor = fprime_max * w
    row = ", ".join(f"q={q}: {factor**q:.3e}" for q in (1, 10, 50, 100))
    print(f"w = {w}: per-step factor {factor:.2f}; {row}")

# The CEC condition from Section 3.2.1: f'(net) * w = 1 with f = identity, w = 1.
print(f"CEC factor f' * w = {1.0 * 1.0}; after q = 1000 steps: {(1.0 * 1.0) ** 1000}")

# Bound from Section 3.1.5: |d theta_v(t-q) / d theta_u(t)| <= n * tau^q with tau = n * w_max / 4.
n, w_max = 2, 1.0
tau = n * w_max / 4.0
print(f"bound with n = {n}, w_max = {w_max}: tau = {tau}")
for q in (10, 50):
    print(f"  q = {q}: n * tau^q = {n * tau**q:.3e}")

# Part 2: Jacobian norms for a tanh RNN, h_t = tanh(W h_{t-1} + U x_t).
rng = np.random.default_rng(0)
hidden, T = 16, 100
W = rng.normal(size=(hidden, hidden)) / np.sqrt(hidden)
U = rng.normal(size=(hidden, 3))
xs = rng.normal(size=(T, 3))
h = np.zeros(hidden)
J = np.eye(hidden)
norms = []
for x in xs:
    h = np.tanh(W @ h + U @ x)
    J = (np.diag(1 - h**2) @ W) @ J
    norms.append(np.linalg.norm(J, 2))
print("spectral radius of W:", round(float(np.max(np.abs(np.linalg.eigvals(W)))), 4))
for t in (1, 10, 25, 50, 100):
    print(f"RNN ||d h_{t} / d h_0||_2 = {norms[t - 1]:.3e}")

# The CEC: s_t = s_{t-1} + (gated input). With truncation (Section 4.5), the Jacobian
# of s_t with respect to s_{t-1} is the identity, so the norm is 1 at every lag.
J = np.eye(hidden)
for t in range(1, T + 1):
    J = np.eye(hidden) @ J
    if t in (1, 10, 25, 50, 100):
        print(f"CEC ||d s_{t} / d s_0||_2 = {np.linalg.norm(J, 2):.3e}")
