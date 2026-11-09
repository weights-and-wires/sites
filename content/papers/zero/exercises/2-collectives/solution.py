import numpy as np


def reduce_scatter(xs):
    """xs has shape (nd, n). Return a list of nd arrays; entry i is chunk i of xs.sum(axis=0)."""
    xs = np.asarray(xs)
    return np.array_split(xs.sum(axis=0), xs.shape[0])


def all_gather(shards):
    """Return a list of len(shards) arrays, each the concatenation of all shards."""
    full = np.concatenate(shards)
    return [full.copy() for _ in shards]


def all_reduce(xs):
    """Return a list of nd arrays, each equal to xs.sum(axis=0). Build it from the two functions above."""
    return all_gather(reduce_scatter(xs))


def comm_volume(stage):
    """Elements moved per process per training step, in units of Psi."""
    if stage == 0:
        return 2  # all-reduce of gradients
    if stage in (1, 2):
        return 1 + 1  # reduce-scatter of gradients, all-gather of parameters
    return 1 + 1 + 1  # reduce-scatter, forward all-gather, backward all-gather
