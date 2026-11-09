import numpy as np


def sample_distractors(masked_idx, K, rng):
    """Return an int array of shape (len(masked_idx), K).

    Row i holds K distinct time steps drawn uniformly from the masked steps other
    than masked_idx[i]. Raise ValueError when fewer than K other masked steps exist.
    """
    raise NotImplementedError


def contrastive_loss(c, q, masked_idx, distractors, kappa):
    """Return the mean contrastive loss of Equation 3 over the masked steps.

    c: (T, d) context vectors. q: (T, d) quantized targets.
    masked_idx: (n,) masked time steps. distractors: (n, K) time steps.
    kappa: temperature. sim is cosine similarity.
    """
    raise NotImplementedError
