import numpy as np


def diversity_loss(logits):
    """Return L_d from Equation 4.

    logits: array of shape (N, G, V). The softmax is over V, with no Gumbel noise
    and no temperature. Average it over N, then return
    sum_g sum_v pbar log pbar / (G * V), using the natural logarithm.
    """
    logits = np.asarray(logits, dtype=float)
    N, G, V = logits.shape
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    pbar = p.mean(axis=0)
    plogp = np.where(pbar > 0, pbar * np.log(np.where(pbar > 0, pbar, 1.0)), 0.0)
    return float(plogp.sum() / (G * V))
