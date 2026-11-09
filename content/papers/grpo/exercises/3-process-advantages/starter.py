import numpy as np


def process_advantages(step_rewards):
    """Process-supervision advantages (Section 4.1.3), one value per step.

    step_rewards: list of G 1-D arrays, the step rewards of each output in order.
    Normalize with the mean and std (ddof=0, plus 1e-6) over ALL steps of ALL outputs,
    then return, for each output, the reverse cumulative sum of its normalized rewards.
    """
    raise NotImplementedError
