import numpy as np


def group_advantages(rewards):
    """Standardize rewards within one group; return zeros when the std is 0."""
    raise NotImplementedError


def keep_group(rewards):
    """True when the group has both a correct (+1) and an incorrect (-1) output."""
    raise NotImplementedError


def fill_batch(groups, batch_size):
    """Read groups until batch_size of them pass keep_group.

    Returns (kept, drawn), where drawn counts every group read.
    """
    raise NotImplementedError
