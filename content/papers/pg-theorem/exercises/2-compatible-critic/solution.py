import numpy as np


def compatible_features(phi, pi):
    """Return psi[s, a] = phi[s, a] - sum_b pi[s, b] phi[s, b], shape (S, A, k)."""
    return phi - np.einsum("sb,sbk->sk", pi, phi)[:, None, :]


def fit_compatible_critic(psi, pi, d, q):
    """Return w, shape (k,), that solves sum_{s,a} d(s) pi(s,a) psi[s,a] (q[s,a] - psi[s,a] @ w) = 0."""
    wts = d[:, None] * pi
    a = np.einsum("sa,sak,sal->kl", wts, psi, psi)
    b = np.einsum("sa,sak,sa->k", wts, psi, q)
    return np.linalg.solve(a, b)


def policy_gradient_with_critic(psi, pi, d, w):
    """Return sum_{s,a} d(s) pi(s,a) psi[s,a] (psi[s,a] @ w), an array of length k (Equation 5)."""
    return np.einsum("sa,sak,sa->k", d[:, None] * pi, psi, psi @ w)
