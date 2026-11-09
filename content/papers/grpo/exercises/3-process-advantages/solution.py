import numpy as np


def process_advantages(step_rewards):
    """Process-supervision advantages (Section 4.1.3), one value per step.

    step_rewards: list of G 1-D arrays, the step rewards of each output in order.
    Normalize with the mean and std (ddof=0, plus 1e-6) over ALL steps of ALL outputs,
    then return, for each output, the reverse cumulative sum of its normalized rewards.
    """
    flat = np.concatenate([np.asarray(s, dtype=float) for s in step_rewards])
    mean, std = flat.mean(), flat.std()
    out = []
    for s in step_rewards:
        norm = (np.asarray(s, dtype=float) - mean) / (std + 1e-6)
        out.append(np.cumsum(norm[::-1])[::-1])
    return out
