import numpy as np

# A softmax policy over K actions in each of S states. The logits are
# phi[s, a] . theta, so phi has shape (S, K, d) and theta has shape (d,).
# rho has shape (S,) and holds the state distribution.


def probs(phi, theta):
    z = phi @ theta
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fisher_softmax(p):
    """Return the Fisher matrix of a tabular softmax policy at one state (Equation 2)."""
    raise NotImplementedError


def fisher(phi, theta, rho):
    """Return F(theta) = sum_s rho(s) E_pi[psi psi^T] (Equations 2 and 3)."""
    raise NotImplementedError
