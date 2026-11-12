import numpy as np


def n_step_target(rewards, done, q_boot, gamma):
    """Return sum_k gamma**k * rewards[k], plus gamma**len(rewards) * q_boot unless done."""
    raise NotImplementedError
