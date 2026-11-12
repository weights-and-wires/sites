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


def fisher(phi, theta, rho):
    """F(theta) from Equations 2 and 3. Finished for you (Exercise 1)."""
    pi = probs(phi, theta)
    psi = scores(phi, theta)
    return np.einsum("s,sk,ska,skb->ab", rho, pi, psi, psi)


def policy_gradient(phi, theta, rho, Q):
    """Equation 1: grad eta = sum_{s,a} rho(s) grad pi(a; s) Q(s, a), with grad pi = pi * psi."""
    pi = probs(phi, theta)
    psi = scores(phi, theta)
    return np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)


def natural_gradient(F, g, lam=1e-3):
    """Equation 4 with damping: solve (F + lam * I) x = g and return x."""
    return np.linalg.solve(F + lam * np.eye(len(g)), g)
