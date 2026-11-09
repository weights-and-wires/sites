import numpy as np


def lm_loss(logits, tokens):
    """Mean next-token negative log-likelihood in nats.

    logits: (T, V); row t scores the token at position t + 1.
    tokens: (T,) integer ids.
    Returns a float averaged over the T - 1 predictions that have a target.
    """
    logits = np.asarray(logits, dtype=float)
    tokens = np.asarray(tokens)
    z = logits[:-1]
    targets = tokens[1:]
    shifted = z - z.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(targets)), targets].mean())


def perplexity(logits, tokens):
    """exp of the mean next-token negative log-likelihood."""
    return float(np.exp(lm_loss(logits, tokens)))
