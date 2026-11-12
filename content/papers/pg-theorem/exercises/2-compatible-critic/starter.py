import numpy as np


def compatible_features(phi, pi):
    """Return psi[s, a] = phi[s, a] - sum_b pi[s, b] phi[s, b], shape (S, A, k)."""
    raise NotImplementedError


def fit_compatible_critic(psi, pi, d, q):
    """Return w, shape (k,), that solves sum_{s,a} d(s) pi(s,a) psi[s,a] (q[s,a] - psi[s,a] @ w) = 0."""
    raise NotImplementedError


def policy_gradient_with_critic(psi, pi, d, w):
    """Return sum_{s,a} d(s) pi(s,a) psi[s,a] (psi[s,a] @ w), an array of length k (Equation 5)."""
    raise NotImplementedError
