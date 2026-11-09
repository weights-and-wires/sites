import numpy as np


def balance(grads, weights, ema, beta=0.999, R=1.0):
    """Return (g_total, new_ema) for the balancer of Equation 5."""
    weights = np.asarray(weights, dtype=float)
    norms = np.array([np.linalg.norm(g) for g in grads])
    new_ema = norms if ema is None else beta * np.asarray(ema) + (1 - beta) * norms
    frac = weights / weights.sum()
    g_total = sum(R * f * g / n for f, g, n in zip(frac, grads, new_ema))
    return g_total, new_ema
