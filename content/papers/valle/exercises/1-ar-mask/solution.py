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
    n = n_text + n_audio
    m = np.zeros((n, n), dtype=bool)
    m[:n_text, :n_text] = True
    m[n_text:, :n_text] = True
    m[n_text:, n_text:] = np.tril(np.ones((n_audio, n_audio), dtype=bool))
    return m


def nar_mask(n_text, n_audio):
    """Return an (n, n) boolean mask for the NAR model: every token sees every token."""
    n = n_text + n_audio
    return np.ones((n, n), dtype=bool)
