import numpy as np


def forward_mask(x0, t, rng, mask_id):
    """Forward masking process (Section 2.1).

    x0: int array of any shape. t: float in [0, 1]. rng: numpy Generator.
    Returns (x_t, masked): x_t is a new array where each token of x0 is
    replaced by mask_id with probability t, independently; masked is the
    boolean array that marks the replaced positions.
    """
    raise NotImplementedError
