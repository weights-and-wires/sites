import numpy as np


def masked_softmax(scores, mask):
    """Finished for you. mask is boolean; True means "may attend"."""
    scores = np.where(mask, scores, -1e9)
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def ar_mask(n_text, n_audio):
    """Return an (n, n) boolean mask for the AR model, with n = n_text + n_audio.

    The first n_text positions are phoneme tokens and the last n_audio
    positions are first-codebook acoustic tokens (prompt, then target).
    """
    raise NotImplementedError


def nar_mask(n_text, n_audio):
    """Return an (n, n) boolean mask for the NAR model: every token sees every token."""
    raise NotImplementedError
