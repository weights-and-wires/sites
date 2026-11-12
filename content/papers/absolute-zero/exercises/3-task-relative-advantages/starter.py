import numpy as np


def task_relative_advantages(rewards, tasks, roles, eps=1e-6):
    """Return (r - mean) / (std + eps) within each (task, role) group.

    rewards: sequence of n floats.
    tasks: sequence of n task labels, such as "ded", "abd", "ind".
    roles: sequence of n role labels, "propose" or "solve".
    Use the population std (ddof=0). Return a float array of shape (n,).
    """
    raise NotImplementedError
