import numpy as np


def mask_tokens(ids, vocab_size, mask_id, rng, mlm_prob=0.15, n_special=4):
    """Corrupt `ids` for masked-language-model training.

    Returns (inputs, labels). `labels` is the original id at selected
    positions and -100 elsewhere. Leaves `ids` unchanged.
    """
    ids = np.asarray(ids)
    selected = (rng.random(ids.shape) < mlm_prob) & (ids >= n_special)
    u = rng.random(ids.shape)
    random_ids = rng.integers(n_special, vocab_size, ids.shape)
    inputs = ids.copy()
    inputs[selected & (u < 0.8)] = mask_id
    swap = selected & (u >= 0.8) & (u < 0.9)
    inputs[swap] = random_ids[swap]
    labels = np.where(selected, ids, -100)
    return inputs, labels
