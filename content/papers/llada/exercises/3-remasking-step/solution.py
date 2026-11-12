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
    masked = x_t == mask_id
    pred = logits.argmax(axis=-1)
    conf = softmax(logits).max(axis=-1)
    conf = np.where(masked, conf, np.inf)
    keep = np.argsort(-conf, kind="stable")[:n_un]
    filled = np.where(masked, pred, x_t)
    out = np.full(len(x_t), mask_id)
    out[keep] = filled[keep]
    return out


def random_step(x_t, logits, mask_id, n_un, rng):
    """One step of random remasking (Algorithm 4), with a fixed count.

    Predict every masked position greedily, then fill exactly
    n_un - (number already unmasked) of them, chosen uniformly at random
    with rng.permutation. Already unmasked tokens never change.
    Returns a new (L,) array.
    """
    masked_idx = np.flatnonzero(x_t == mask_id)
    n_new = n_un - (len(x_t) - len(masked_idx))
    chosen = rng.permutation(masked_idx)[:n_new]
    out = x_t.copy()
    out[chosen] = logits.argmax(axis=-1)[chosen]
    return out
