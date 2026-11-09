import numpy as np


def optimal_discriminator(p_data, p_g):
    """Return D*(x) = p_data / (p_data + p_g); return 0.5 where both are 0."""
    p_data = np.asarray(p_data, dtype=float)
    p_g = np.asarray(p_g, dtype=float)
    denom = p_data + p_g
    out = np.full_like(denom, 0.5)
    np.divide(p_data, denom, out=out, where=denom > 0)
    return out


def value(p_data, p_g, D):
    """Return V(G, D) = sum p_data log D + p_g log(1 - D), with 0 log 0 = 0."""
    p_data = np.asarray(p_data, dtype=float)
    p_g = np.asarray(p_g, dtype=float)
    D = np.asarray(D, dtype=float)
    t1 = np.where(p_data > 0, p_data * np.log(np.where(p_data > 0, D, 1.0)), 0.0)
    t2 = np.where(p_g > 0, p_g * np.log(np.where(p_g > 0, 1.0 - D, 1.0)), 0.0)
    return float(np.sum(t1 + t2))


def _kl(p, q):
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def jsd(p, q):
    """Return the Jensen-Shannon divergence (natural log) of two probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = (p + q) / 2
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def virtual_criterion(p_data, p_g):
    """Return C(G) = V(G, D*_G)."""
    return value(p_data, p_g, optimal_discriminator(p_data, p_g))
