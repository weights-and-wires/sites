import numpy as np


def _assign(codebook, x):
    d = ((x[:, None, :] - codebook[None]) ** 2).sum(-1)
    return d.argmin(axis=1)


def ema_step(codebook, ema_count, ema_sum, batch, rng, decay=0.99, threshold=0.5, eps=1e-5):
    """One EMA update of a codebook, then replace dead entries from the batch.

    Returns (codebook, ema_count, ema_sum, idx). Does not modify its inputs.
    """
    K, D = codebook.shape
    idx = _assign(codebook, batch)
    counts = np.bincount(idx, minlength=K).astype(float)
    sums = np.zeros((K, D))
    np.add.at(sums, idx, batch)
    ema_count = decay * ema_count + (1 - decay) * counts
    ema_sum = decay * ema_sum + (1 - decay) * sums
    codebook = ema_sum / np.maximum(ema_count, eps)[:, None]
    for k in np.flatnonzero(ema_count < threshold):
        v = batch[rng.integers(len(batch))]
        codebook[k] = v
        ema_sum[k] = v
        ema_count[k] = 1.0
    return codebook, ema_count, ema_sum, idx


def usage(codebook, x):
    """Return (n_used, perplexity) of the nearest-entry assignments of x."""
    idx = _assign(codebook, x)
    p = np.bincount(idx, minlength=len(codebook)) / len(idx)
    p = p[p > 0]
    return int(len(p)), float(np.exp(-(p * np.log(p)).sum()))
