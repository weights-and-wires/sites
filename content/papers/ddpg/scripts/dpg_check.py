"""Numbers for the "Deterministic policy gradient" example (Equation 6).

A linear actor mu(s) = theta . s and a quadratic critic Q(s, a) = -(a - k . s)^2.
The chain rule gives grad_a Q * grad_theta mu = -2 (mu(s) - k . s) * s.

Run: python3 scripts/dpg_check.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

k = np.array([1.0, -0.5])
theta = np.array([0.2, 0.4])
S = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def q(s, a):
    return -((a - s @ k) ** 2)


def j(theta):
    return np.mean(q(S, S @ theta))


a = S @ theta
dq_da = -2 * (a - S @ k)
grad = np.mean(dq_da[:, None] * S, axis=0)  # grad_theta mu(s) = s for each s

print("mu(s)         =", a)
print("k . s         =", S @ k)
print("grad_a Q      =", dq_da)
print("per-sample    =\n", dq_da[:, None] * S)
print("DPG gradient  =", grad)

eps = 1e-6
fd = np.array([(j(theta + eps * e) - j(theta - eps * e)) / (2 * eps) for e in np.eye(2)])
print("finite diff   =", fd)
print(f"max abs diff  = {np.abs(fd - grad).max():.2e}")

# The wrong gradient that forgets grad_theta mu (drops the factor s).
print("without chain-rule factor s:", np.mean(dq_da))

# Gradient ascent with learning rate 0.1.
th = theta.copy()
print(f"J at start = {j(th):.4f}")
for step in range(300):
    a = S @ th
    th = th + 0.1 * np.mean((-2 * (a - S @ k))[:, None] * S, axis=0)
    if step + 1 in (1, 10, 100):
        print(f"after {step + 1:>3} steps: theta = {th}, J = {j(th):.5f}")
print("after 300 steps: theta =", th, f"J = {j(th):.2e}")
