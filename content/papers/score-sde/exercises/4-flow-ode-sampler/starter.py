import numpy as np

BMIN, BMAX = 0.1, 20.0
W = np.array([0.5, 0.3, 0.2])
MU = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
S = 0.3


def beta(t):
    """Finished for you. Appendix B, Equation 32."""
    return BMIN + t * (BMAX - BMIN)


def int_beta(t):
    """Finished for you. Integral of beta from 0 to t."""
    return BMIN * t + 0.5 * (BMAX - BMIN) * t**2


def toy_score(x, t):
    """Finished for you. Exact score of p_t for the three-mode mixture, x of shape (n, 2)."""
    m = np.exp(-0.5 * int_beta(t))
    v = m**2 * S**2 + 1.0 - m**2
    d = x[:, None, :] - m * MU[None]
    logw = np.log(W)[None] - 0.5 * np.sum(d**2, axis=-1) / v
    r = np.exp(logw - logw.max(axis=1, keepdims=True))
    r /= r.sum(axis=1, keepdims=True)
    return -np.sum(r[:, :, None] * d, axis=1) / v


def pf_ode_sample(score_fn, x_T, n_steps, eps=1e-3):
    """Integrate Equation 13 from t = 1 down to t = eps with Euler steps.

    x_T has shape (n, 2). Returns an array of the same shape.
    """
    raise NotImplementedError
