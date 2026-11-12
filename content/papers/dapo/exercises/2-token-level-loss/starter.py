import numpy as np


def objective(terms, level):
    """Reduce a list of per-token term arrays to one number.

    level "sample": mean over outputs of the mean over tokens (Equation 5).
    level "token": sum of all terms divided by the total token count (Equation 12).
    """
    raise NotImplementedError


def token_weights(lengths, level):
    """Return one array per output with the weight each of its tokens gets."""
    raise NotImplementedError
