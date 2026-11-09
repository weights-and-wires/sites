import numpy as np

# A softmax policy over K actions for each of N sampled states. The logits are
# phi[n, k] . theta, so phi has shape (N, K, d) and theta has shape (d,).


def probs(phi, theta):
    z = phi @ theta
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def conjugate_gradient(fvp, b, iters=10, tol=1e-10):
    """Finished for you (Exercise 3)."""
    x = np.zeros_like(b, dtype=float)
    r = b.astype(float).copy()
    p = r.copy()
    rs = r @ r
    for _ in range(iters):
        if rs < tol:
            break
        Ap = fvp(p)
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def fisher_vector_product(phi, theta, v):
    """Return F v without forming F, where F is the Hessian of the mean KL at theta (Appendix C.1).

    For each state, center the features: c[k] = phi[k] - sum_j p[j] phi[j]. Then
    F = mean over states of sum_k p[k] c[k] c[k]^T, so F v = mean of sum_k p[k] c[k] (c[k] . v).
    """
    raise NotImplementedError


def trpo_step(theta_old, g, fvp, delta, surrogate, kl, shrink=0.5, max_backtracks=10):
    """One TRPO update: natural gradient direction, maximal step, then a backtracking line search.

    1. s = A^-1 g by conjugate gradient with 10 iterations, where fvp(v) = A v.
    2. beta = sqrt(2 * delta / (s^T A s)).
    3. For k = 0, 1, ..., max_backtracks - 1, try theta = theta_old + shrink**k * beta * s.
       Return the first theta with kl(theta) <= delta and surrogate(theta) > surrogate(theta_old).
    4. If no candidate qualifies, return theta_old.

    `surrogate(theta)` and `kl(theta)` are functions of the new parameters. kl(theta) is the
    mean KL divergence from the old policy to the policy at theta.
    """
    raise NotImplementedError
