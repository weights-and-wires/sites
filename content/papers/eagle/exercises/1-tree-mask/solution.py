import numpy as np


def tree_depths(parents):
    """Finished for you: depth of each node. A root has depth 1."""
    depth = [0] * len(parents)
    for i, p in enumerate(parents):
        depth[i] = 1 if p < 0 else depth[p] + 1
    return np.array(depth, dtype=int)


def position_ids(parents, ctx_len):
    """Return an int array of shape (n,): ctx_len + depth - 1 for each node."""
    return ctx_len + tree_depths(parents) - 1


def tree_mask(parents, ctx_len):
    """Return a bool array of shape (n, ctx_len + n).

    mask[i, j] is True for every context column, for node i itself, and for
    every ancestor of node i. It is False everywhere else.
    """
    n = len(parents)
    mask = np.zeros((n, ctx_len + n), dtype=bool)
    mask[:, :ctx_len] = True
    for i in range(n):
        j = i
        while j >= 0:
            mask[i, ctx_len + j] = True
            j = parents[j]
    return mask
