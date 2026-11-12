"""Numbers for the "Policy loss" example (Section 4.2 Equations 4, 12, 13).

One state, a ~ N(mu, sigma^2) with sigma = exp(s), Q(a) = -(a - 2)^2, temperature 1.
J_pi = E[log pi(a) - Q(a)] with a = mu + sigma eps. Since exp(Q) is proportional to N(2, 1/2),
Equation 4 says the best Gaussian has mu = 2 and sigma = sqrt(1/2).

Run: python3 scripts/policy_loss.py
"""

import numpy as np


def q(a):
    return -((a - 2.0) ** 2)


def dq(a):
    return -2.0 * (a - 2.0)


def loss(mu, s, eps):
    sigma = np.exp(s)
    a = mu + sigma * eps
    logp = -0.5 * eps**2 - s - 0.5 * np.log(2 * np.pi)  # log N(a; mu, sigma) at a = f(eps)
    return np.mean(logp - q(a))


def grads(mu, s, eps):
    sigma = np.exp(s)
    a = mu + sigma * eps
    g_mu = -np.mean(dq(a))                 # Equation 13: the log pi terms cancel for mu
    g_s = -1.0 - sigma * np.mean(dq(a) * eps)  # d/ds, with d sigma / d s = sigma
    return g_mu, g_s


eps = np.array([-1.0, 0.0, 1.0, 0.5])
mu, s = 0.5, 0.0
print(f"mu = {mu}, log sigma = {s}, eps = {eps}")
print(f"J_pi = {loss(mu, s, eps):.5f}")
g_mu, g_s = grads(mu, s, eps)
print(f"analytic grads: dJ/dmu = {g_mu:.5f}, dJ/dlog_sigma = {g_s:.5f}")
h = 1e-6
fd_mu = (loss(mu + h, s, eps) - loss(mu - h, s, eps)) / (2 * h)
fd_s = (loss(mu, s + h, eps) - loss(mu, s - h, eps)) / (2 * h)
print(f"finite differences: {fd_mu:.5f}, {fd_s:.5f}")

# Gradient descent with a fixed set of 200000 noise draws reaches the Equation 4 optimum.
rng = np.random.default_rng(0)
big = rng.standard_normal(200000)
mu, s = 0.5, 0.0
for _ in range(2000):
    g_mu, g_s = grads(mu, s, big)
    mu -= 0.05 * g_mu
    s -= 0.05 * g_s
print(f"after 2000 steps: mu = {mu:.3f}, sigma = {np.exp(s):.3f}  (target mu = 2, sigma = {np.sqrt(0.5):.3f})")

# Without the entropy term the loss would push sigma to 0.
mu, s = 0.5, 0.0
for _ in range(2000):
    a = mu + np.exp(s) * big
    mu -= 0.05 * (-np.mean(dq(a)))
    s -= 0.05 * (-np.exp(s) * np.mean(dq(a) * big))
print(f"without the entropy bonus: mu = {mu:.3f}, sigma = {np.exp(s):.4f}")
