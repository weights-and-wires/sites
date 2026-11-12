import numpy as np


def mask_tokens(ids, vocab_size, mask_id, rng, mlm_prob=0.15, n_special=4):
    """Corrupt `ids` for masked-language-model training.

    Returns (inputs, labels). `labels` is the original id at selected
    positions and -100 elsewhere. Leaves `ids` unchanged.
    """
    raise NotImplementedError
