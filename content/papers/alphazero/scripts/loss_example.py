"""Numbers for the "One network, two heads, one loss" example.

The AlphaZero loss (Equation 1 of the paper) is
    l = (z - v)^2 - pi^T log p + c * ||theta||^2.
This script evaluates it on a four-move toy position and checks the gradient
with respect to the logits and the value against finite differences.

Run: python3 scripts/loss_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def az_loss(logits, v, pi, z, theta_sq_norm=0.0, c=0.0):
    p = softmax(logits)
    value_term = (z - v) ** 2
    policy_term = -float(pi @ np.log(p))
    return value_term + policy_term + c * theta_sq_norm, value_term, policy_term


def grads(logits, v, pi, z):
    """Analytic gradients: d l / d logits = p - pi (pi sums to 1), d l / d v = -2 (z - v)."""
    return softmax(logits) - pi, -2.0 * (z - v)


if __name__ == "__main__":
    logits = np.array([1.0, 0.5, 0.0, -1.0])
    pi = np.array([0.6, 0.3, 0.1, 0.0])  # search visit shares, from the visit_targets example
    v, z = 0.3, -1.0
    c, theta_sq = 1e-4, 50.0

    p = softmax(logits)
    total, value_term, policy_term = az_loss(logits, v, pi, z, theta_sq, c)
    print(f"p = {p}")
    print(f"value term  (z - v)^2      = {value_term:.4f}")
    print(f"policy term -pi^T log p    = {policy_term:.4f}")
    print(f"L2 term     c ||theta||^2  = {c * theta_sq:.4f}")
    print(f"total loss l               = {total:.4f}")

    entropy = -float(pi[pi > 0] @ np.log(pi[pi > 0]))
    at_target = az_loss(np.log(np.where(pi > 0, pi, 1e-12)), z, pi, z)[2]
    print(f"entropy of pi              = {entropy:.4f}")
    print(f"policy term when p == pi   = {at_target:.4f}  (the floor is the entropy, not 0)")

    g_logits, g_v = grads(logits, v, pi, z)
    print(f"analytic  d l / d logits   = {g_logits}")
    print(f"analytic  d l / d v        = {g_v:.4f}")
    eps = 1e-6
    fd = np.zeros(4)
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        fd[i] = (az_loss(logits + d, v, pi, z)[0] - az_loss(logits - d, v, pi, z)[0]) / (2 * eps)
    fd_v = (az_loss(logits, v + eps, pi, z)[0] - az_loss(logits, v - eps, pi, z)[0]) / (2 * eps)
    print(f"numeric   d l / d logits   = {fd}")
    print(f"numeric   d l / d v        = {fd_v:.4f}")
