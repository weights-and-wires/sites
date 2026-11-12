import numpy as np


def tree_depths(parents):
    """Finished for you: depth of each node. A root has depth 1."""
    depth = [0] * len(parents)
    for i, p in enumerate(parents):
        depth[i] = 1 if p < 0 else depth[p] + 1
    return np.array(depth, dtype=int)


def position_ids(parents, ctx_len):
    """Return an int array of shape (n,): ctx_len + depth - 1 for each node."""
    raise NotImplementedError


def tree_mask(parents, ctx_len):
    """Return a bool array of shape (n, ctx_len + n).

    mask[i, j] is True for every context column, for node i itself, and for
    every ancestor of node i. It is False everywhere else.
    """
    raise NotImplementedError
