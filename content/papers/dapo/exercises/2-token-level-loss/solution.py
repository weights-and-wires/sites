import numpy as np


def objective(terms, level):
    """Reduce a list of per-token term arrays to one number.

    level "sample": mean over outputs of the mean over tokens (Equation 5).
    level "token": sum of all terms divided by the total token count (Equation 12).
    """
    if level == "sample":
        return float(np.mean([np.mean(t) for t in terms]))
    if level == "token":
        return float(sum(np.sum(t) for t in terms) / sum(len(t) for t in terms))
    raise ValueError(f"unknown level: {level!r}")


def token_weights(lengths, level):
    """Return one array per output with the weight each of its tokens gets."""
    g = len(lengths)
    total = sum(lengths)
    if level == "sample":
        return [np.full(n, 1.0 / (g * n)) for n in lengths]
    if level == "token":
        return [np.full(n, 1.0 / total) for n in lengths]
    raise ValueError(f"unknown level: {level!r}")
