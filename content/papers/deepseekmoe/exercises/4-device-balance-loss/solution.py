import numpy as np


def device_balance_loss(f, P, groups, alpha=1.0):
    """Return alpha * sum_i f'_i * P'_i (Equations 15-17)."""
    f = np.asarray(f, dtype=float)
    P = np.asarray(P, dtype=float)
    total = 0.0
    for g in groups:
        total += f[g].mean() * P[g].sum()
    return float(alpha * total)
