import numpy as np


def ema_step(codebook, ema_count, ema_sum, batch, rng, decay=0.99, threshold=0.5, eps=1e-5):
    """One EMA update of a codebook, then replace dead entries from the batch.

    Returns (codebook, ema_count, ema_sum, idx). Does not modify its inputs.
    """
    raise NotImplementedError


def usage(codebook, x):
    """Return (n_used, perplexity) of the nearest-entry assignments of x."""
    raise NotImplementedError
