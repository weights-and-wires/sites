import numpy as np


def score(option, mode):
    """Score one completion. option: {"ctx": [...], "unc": [...] or None} of per-token log-probs."""
    raise NotImplementedError


def choose(options, mode):
    """Index of the best option under the given scoring mode."""
    raise NotImplementedError
