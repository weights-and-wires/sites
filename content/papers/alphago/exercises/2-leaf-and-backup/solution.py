import numpy as np


def leaf_value(v, z, lam):
    """Return (1 - lam) * v + lam * z."""
    return (1.0 - lam) * v + lam * z


def backup(N, W, path, v_leaf):
    """Update N and W in place along path (root edge first).

    v_leaf is the value for the player to move at the leaf. The last edge on the
    path gets -v_leaf, the one before it gets +v_leaf, and so on. Each edge on the
    path adds 1 to N.
    """
    sign = -1.0
    for e in reversed(path):
        N[e] += 1
        W[e] += sign * v_leaf
        sign = -sign


def mixed_q(Wv, Nv, Wr, Nr, lam):
    """Return (1 - lam) * Wv / Nv + lam * Wr / Nr, with a term of 0 when its count is 0."""
    qv = Wv / Nv if Nv > 0 else 0.0
    qr = Wr / Nr if Nr > 0 else 0.0
    return (1.0 - lam) * qv + lam * qr
