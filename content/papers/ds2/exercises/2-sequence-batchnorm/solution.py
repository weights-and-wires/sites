import numpy as np


def sequence_batchnorm(a, lengths, gamma, beta, eps=1e-5):
    """Sequence-wise batch normalization of a (B, T, d) tensor of pre-activations W h_t.

    Statistics are per unit, over every valid (utterance, time step) pair.
    Padded positions come back as 0.
    """
    a = np.asarray(a, dtype=float)
    B, T, d = a.shape
    mask = np.arange(T)[None, :] < np.asarray(lengths)[:, None]  # (B, T)
    valid = a[mask]  # (N, d)
    mean = valid.mean(axis=0)
    var = valid.var(axis=0)
    out = gamma * (a - mean) / np.sqrt(var + eps) + beta
    return np.where(mask[:, :, None], out, 0.0)


def framewise_batchnorm(a, gamma, beta, eps=1e-5):
    """Frame-wise batch normalization of a (B, T, d) tensor: statistics per time step and unit, over the batch only."""
    a = np.asarray(a, dtype=float)
    mean = a.mean(axis=0, keepdims=True)
    var = a.var(axis=0, keepdims=True)
    return gamma * (a - mean) / np.sqrt(var + eps) + beta
