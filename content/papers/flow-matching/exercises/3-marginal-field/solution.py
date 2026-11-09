import numpy as np


def marginal_vf(x, t, data, sigma_min):
    """Equation 8 for the empirical distribution on `data` and the OT path.

    x: (n, d). t: float in [0, 1]. data: (N, d). Returns (n, d).
    """
    sigma = 1.0 - (1.0 - sigma_min) * t
    d2 = np.sum((x[:, None, :] - t * data[None]) ** 2, axis=-1)  # (n, N)
    logw = -0.5 * d2 / sigma**2
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    u = (data[None] - (1.0 - sigma_min) * x[:, None, :]) / sigma  # (n, N, d)
    return np.sum(w[:, :, None] * u, axis=1)
