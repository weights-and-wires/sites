"""Shared helpers for the pg-theorem page: a tabular MDP with a softmax policy.

Conventions (start-state formulation, Section 1 of the paper):
  P[s, a, s2]  transition probabilities
  R[s, a]      expected reward for taking a in s
  theta[s, a]  softmax logits, so pi(s, a) = exp(theta[s, a]) / sum_b exp(theta[s, b])
  d(s)         sum_t gamma^t Pr(s_t = s | s_0, pi), NOT normalized: it sums to 1 / (1 - gamma)
The paper calls the objective rho; this page and the code call it J.
"""

import numpy as np


def softmax_rows(theta):
    z = theta - theta.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def policy_matrices(P, R, pi):
    p_pi = np.einsum("sa,sat->st", pi, P)
    r_pi = np.einsum("sa,sa->s", pi, R)
    return p_pi, r_pi


def state_values(P, R, pi, gamma):
    p_pi, r_pi = policy_matrices(P, R, pi)
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * p_pi, r_pi)


def q_values(P, R, pi, gamma):
    return R + gamma * np.einsum("sat,t->sa", P, state_values(P, R, pi, gamma))


def state_weights(P, pi, gamma, s0):
    p_pi = np.einsum("sa,sat->st", pi, P)
    e0 = np.zeros(P.shape[0])
    e0[s0] = 1.0
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * p_pi.T, e0)


def objective(P, R, theta, gamma, s0):
    return float(state_values(P, R, softmax_rows(theta), gamma)[s0])


def policy_gradient(P, R, theta, gamma, s0):
    """Theorem 1 (Equation 2) for a tabular softmax policy."""
    pi = softmax_rows(theta)
    q = q_values(P, R, pi, gamma)
    d = state_weights(P, pi, gamma, s0)
    # sum_a dpi(s,a)/dtheta[s,b] Q(s,a) = pi(s,b) * (Q(s,b) - sum_a pi(s,a) Q(s,a))
    v = (pi * q).sum(axis=1, keepdims=True)
    return d[:, None] * pi * (q - v)


def finite_difference_gradient(P, R, theta, gamma, s0, h=1e-6):
    g = np.zeros_like(theta)
    for idx in np.ndindex(*theta.shape):
        up, dn = theta.copy(), theta.copy()
        up[idx] += h
        dn[idx] -= h
        g[idx] = (objective(P, R, up, gamma, s0) - objective(P, R, dn, gamma, s0)) / (2 * h)
    return g


def two_state_mdp():
    """The by-hand example on the page: deterministic, 2 states, 2 actions."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = P[0, 1, 1] = P[1, 0, 1] = P[1, 1, 0] = 1.0
    R = np.array([[0.0, 0.0], [1.0, 0.0]])
    return P, R


def random_mdp(n_states=3, n_actions=2, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((n_states, n_actions, n_states)) ** 2
    P /= P.sum(axis=2, keepdims=True)
    R = rng.normal(size=(n_states, n_actions))
    return P, R


def compatible_features(phi, pi):
    """psi(s, a) = phi(s, a) - sum_b pi(s, b) phi(s, b) (Section 3)."""
    return phi - np.einsum("sb,sbk->sk", pi, phi)[:, None, :]


def linear_softmax_pi(phi, theta):
    logits = phi @ theta
    return softmax_rows(logits)


def fit_w(psi, pi, d, q):
    """Solve Equation 3 for linear f_w = psi . w: weighted least squares."""
    wts = d[:, None] * pi
    a = np.einsum("sa,sak,sal->kl", wts, psi, psi)
    b = np.einsum("sa,sak,sa->k", wts, psi, q)
    return np.linalg.solve(a, b)
