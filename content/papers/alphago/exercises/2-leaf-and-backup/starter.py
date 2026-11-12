import numpy as np


def leaf_value(v, z, lam):
    """Return (1 - lam) * v + lam * z."""
    raise NotImplementedError


def backup(N, W, path, v_leaf):
    """Update N and W in place along path (root edge first).

    v_leaf is the value for the player to move at the leaf. The last edge on the
    path gets -v_leaf, the one before it gets +v_leaf, and so on. Each edge on the
    path adds 1 to N.
    """
    raise NotImplementedError


def mixed_q(Wv, Nv, Wr, Nr, lam):
    """Return (1 - lam) * Wv / Nv + lam * Wr / Nr, with a term of 0 when its count is 0."""
    raise NotImplementedError
