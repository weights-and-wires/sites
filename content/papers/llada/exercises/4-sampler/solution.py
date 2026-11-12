import numpy as np


def softmax(z):
    """Finished for you. Softmax over the last axis."""
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def lowconf_step(x_t, logits, mask_id, n_un):
    """Finished for you. One step of low-confidence remasking (Algorithm 5).

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
    """Finished for you. One step of random remasking (Algorithm 4), with a fixed count.

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

def generate(predict, length, steps, mask_id, remask="low_confidence", rng=None):
    """The reverse process of Section 2.4 with N = steps uniformly spaced steps.

    predict(x) takes the current (length,) int array and returns (length, V)
    logits. Start from a fully masked sequence. At step k = 1, ..., steps, call
    predict once on the current sequence, then apply one remasking step so
    that n_un = length * k // steps positions are unmasked afterward. Use
    lowconf_step when remask is "low_confidence" and random_step when it is
    "random". Raise ValueError for any other strategy.
    Returns the final (length,) array, which has no masked positions.
    """
    if remask not in ("low_confidence", "random"):
        raise ValueError(f"unknown remasking strategy: {remask!r}")
    x = np.full(length, mask_id)
    for k in range(1, steps + 1):
        logits = predict(x)
        n_un = length * k // steps
        if remask == "low_confidence":
            x = lowconf_step(x, logits, mask_id, n_un)
        else:
            x = random_step(x, logits, mask_id, n_un, rng)
    return x
