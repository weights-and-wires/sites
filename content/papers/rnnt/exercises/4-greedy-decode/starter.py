import numpy as np


def greedy_decode(joint, T, max_symbols=3):
    """Greedy transducer decoding.

    joint(t, prefix) -> probabilities over (null, label 1, ..., label K),
    where t is the 0-based frame and prefix is a tuple of emitted labels.
    Returns a tuple of labels.
    """
    raise NotImplementedError
