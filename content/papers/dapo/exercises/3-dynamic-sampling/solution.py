import numpy as np


def group_advantages(rewards):
    """Standardize rewards within one group; return zeros when the std is 0."""
    r = np.asarray(rewards, dtype=float)
    std = r.std()
    if std == 0:
        return np.zeros_like(r)
    return (r - r.mean()) / std


def keep_group(rewards):
    """True when the group has both a correct (+1) and an incorrect (-1) output."""
    r = np.asarray(rewards)
    correct = int(np.sum(r == 1))
    return 0 < correct < len(r)


def fill_batch(groups, batch_size):
    """Read groups until batch_size of them pass keep_group.

    Returns (kept, drawn), where drawn counts every group read.
    """
    kept = []
    drawn = 0
    if batch_size <= 0:
        return kept, drawn
    for g in groups:
        drawn += 1
        if keep_group(g):
            kept.append(g)
            if len(kept) == batch_size:
                break
    return kept, drawn
