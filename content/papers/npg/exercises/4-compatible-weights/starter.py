import numpy as np

# A softmax policy over K actions in each of S states. The logits are
# phi[s, a] . theta, so phi has shape (S, K, d) and theta has shape (d,).
# rho has shape (S,) and holds the state distribution. Q has shape (S, K).


def probs(phi, theta):
    z = phi @ theta
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def scores(phi, theta):
    """psi(s, a) = grad_theta log pi(a; s, theta), shape (S, K, d). Given."""
    pi = probs(phi, theta)
    return phi - np.einsum("sk,skd->sd", pi, phi)[:, None, :]


def compatible_weights(phi, theta, rho, Q):
    """Return the w that minimizes sum_{s,a} rho(s) pi(a; s) (psi(s, a) . w - Q(s, a))^2 (Section 3.1)."""
    raise NotImplementedError
