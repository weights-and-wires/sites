import numpy as np


def conjugate_gradient(fvp, b, iters=10, tol=1e-10):
    """Approximately solve A x = b using only the function fvp(v) = A v.

    Start from x = 0. Stop after `iters` iterations, or earlier when the squared
    residual norm r.r drops below `tol`. Return x.
    """
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


def max_step_length(s, As, delta):
    """beta = sqrt(2 * delta / (s^T A s)), given s and the product As = A s (Appendix C)."""
    return np.sqrt(2 * delta / (s @ As))
