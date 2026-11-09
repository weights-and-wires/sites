import numpy as np


def reduce_scatter(xs):
    """xs has shape (nd, n). Return a list of nd arrays; entry i is chunk i of xs.sum(axis=0)."""
    raise NotImplementedError


def all_gather(shards):
    """Return a list of len(shards) arrays, each the concatenation of all shards."""
    raise NotImplementedError


def all_reduce(xs):
    """Return a list of nd arrays, each equal to xs.sum(axis=0). Build it from the two functions above."""
    raise NotImplementedError


def comm_volume(stage):
    """Elements moved per process per training step, in units of Psi."""
    raise NotImplementedError
