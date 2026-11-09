import numpy as np


def repeat_rates(idx):
    """Return (first, either): repetition rates of consecutive tokens' expert choices."""
    idx = np.asarray(idx)
    first = float(np.mean(idx[1:, 0] == idx[:-1, 0]))
    overlap = (idx[1:, :, None] == idx[:-1, None, :]).any(axis=(1, 2))
    return first, float(np.mean(overlap))
