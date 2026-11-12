import numpy as np


def task_relative_advantages(rewards, tasks, roles, eps=1e-6):
    """Return (r - mean) / (std + eps) within each (task, role) group.

    rewards: sequence of n floats.
    tasks: sequence of n task labels, such as "ded", "abd", "ind".
    roles: sequence of n role labels, "propose" or "solve".
    Use the population std (ddof=0). Return a float array of shape (n,).
    """
    rewards = np.asarray(rewards, dtype=float)
    out = np.zeros_like(rewards)
    keys = list(zip(tasks, roles))
    for key in set(keys):
        mask = np.array([k == key for k in keys])
        group = rewards[mask]
        out[mask] = (group - group.mean()) / (group.std() + eps)
    return out
