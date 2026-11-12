import numpy as np


def last_token_pool(hidden, attention_mask):
    """Return hidden[b, last real position of row b] for every row, shape (B, d)."""
    hidden = np.asarray(hidden)
    mask = np.asarray(attention_mask)
    B, T = mask.shape
    idx = T - 1 - np.argmax(mask[:, ::-1], axis=1)
    return hidden[np.arange(B), idx]


def embed(hidden, attention_mask):
    """Return the pooled vectors with unit L2 norm, shape (B, d)."""
    pooled = last_token_pool(hidden, attention_mask)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
