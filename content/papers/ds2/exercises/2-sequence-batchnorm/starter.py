import numpy as np


def sequence_batchnorm(a, lengths, gamma, beta, eps=1e-5):
    """Sequence-wise batch normalization of a (B, T, d) tensor of pre-activations W h_t.

    Statistics are per unit, over every valid (utterance, time step) pair.
    Padded positions come back as 0.
    """
    raise NotImplementedError


def framewise_batchnorm(a, gamma, beta, eps=1e-5):
    """Frame-wise batch normalization of a (B, T, d) tensor: statistics per time step and unit, over the batch only."""
    raise NotImplementedError
