import numpy as np


def last_token_pool(hidden, attention_mask):
    """Return hidden[b, last real position of row b] for every row, shape (B, d)."""
    raise NotImplementedError


def embed(hidden, attention_mask):
    """Return the pooled vectors with unit L2 norm, shape (B, d)."""
    raise NotImplementedError
