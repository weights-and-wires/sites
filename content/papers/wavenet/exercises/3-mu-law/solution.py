import numpy as np


def mu_law_encode(x, mu=255):
    """Map floats in [-1, 1] to integers in 0..mu."""
    x = np.clip(np.asarray(x, dtype=float), -1.0, 1.0)
    f = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((f + 1) / 2 * mu + 0.5).astype(int)


def mu_law_decode(q, mu=255):
    """Map integers in 0..mu back to floats in [-1, 1]."""
    f = 2.0 * np.asarray(q, dtype=float) / mu - 1
    return np.sign(f) * np.expm1(np.abs(f) * np.log1p(mu)) / mu
