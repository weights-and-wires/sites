import numpy as np


def speaker_posterior(prompt, trans, prior):
    """Return p(k | prompt) as an array of shape (K,).

    prompt: list of ints. trans: (K, V, V), rows sum to 1. prior: (K,).
    """
    raise NotImplementedError


def next_token_dist(prefix, trans, prior):
    """Return p(next token | prefix) as an array of shape (V,), mixing over speakers."""
    raise NotImplementedError


def continue_greedy(prefix, n, trans, prior):
    """Return prefix + n greedy tokens, as a list of ints."""
    raise NotImplementedError
