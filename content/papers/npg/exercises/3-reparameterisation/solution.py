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
    """Equation 1. Finished for you (Exercise 2)."""
    pi = probs(phi, theta)
    psi = scores(phi, theta)
    return np.einsum("s,sk,skd,sk->d", rho, pi, psi, Q)


def scaled_step(phi, theta, rho, Q, c, alpha, natural):
    """Take one step in the coordinates u = theta / c and return the change in theta.

    The policy is the same policy: its logits are theta . phi = u . (c * phi), because theta = c * u.
    Compute the gradient g_u and the Fisher matrix F_u in the u coordinates, then
      natural=False: delta_u = alpha * g_u
      natural=True:  delta_u = alpha * solve(F_u, g_u)
    Return delta_theta = c * delta_u, so the caller can compare steps in the original coordinates.
    """
    c = np.asarray(c, dtype=float)
    phi_u = phi * c
    u = theta / c
    g_u = policy_gradient(phi_u, u, rho, Q)
    if natural:
        delta_u = alpha * np.linalg.solve(fisher(phi_u, u, rho), g_u)
    else:
        delta_u = alpha * g_u
    return c * delta_u
