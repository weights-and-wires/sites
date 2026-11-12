import numpy as np


def n_step_target(rewards, done, q_boot, gamma):
    """Return sum_k gamma**k * rewards[k], plus gamma**len(rewards) * q_boot unless done."""
    m = len(rewards)
    ret = sum(gamma**k * r for k, r in enumerate(rewards))
    if done:
        return float(ret)
    return float(ret + gamma**m * q_boot)
