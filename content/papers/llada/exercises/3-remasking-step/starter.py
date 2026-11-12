import numpy as np


def softmax(z):
    """Finished for you. Softmax over the last axis."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def lowconf_step(x_t, logits, mask_id, n_un):
    """One step of low-confidence remasking (Algorithm 5).

    x_t: (L,) int array; mask_id marks the masked positions.
    logits: (L, V) float array from the mask predictor on x_t.
    n_un: number of unmasked positions the result must have. Assume it is at
    least the number already unmasked in x_t.
    Predict every masked position greedily. The confidence of a prediction is
    its softmax probability. Positions that are already unmasked count as
    infinitely confident. Keep the n_un most confident positions, and remask
    the rest. Break ties toward the lower index.
    Returns a new (L,) array.
    """
    raise NotImplementedError


def random_step(x_t, logits, mask_id, n_un, rng):
    """One step of random remasking (Algorithm 4), with a fixed count.

    Predict every masked position greedily, then fill exactly
    n_un - (number already unmasked) of them, chosen uniformly at random
    with rng.permutation. Already unmasked tokens never change.
    Returns a new (L,) array.
    """
    raise NotImplementedError
