import numpy as np


def lm_loss(logits, tokens):
    """Mean next-token negative log-likelihood in nats.

    logits: (T, V); row t scores the token at position t + 1.
    tokens: (T,) integer ids.
    Returns a float averaged over the T - 1 predictions that have a target.
    """
    raise NotImplementedError


def perplexity(logits, tokens):
    """exp of the mean next-token negative log-likelihood."""
    raise NotImplementedError
