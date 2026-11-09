import numpy as np


def sample_probs(logits, temperature=1.0, top_k=None):
    """Probabilities to sample from, after temperature and top-k.

    logits: (V,) array. temperature: positive float. top_k: None or an int >= 1.
    Returns a (V,) array that sums to 1, with zeros outside the top k.
    """
    z = np.asarray(logits, dtype=float) / temperature
    if top_k is not None and top_k < len(z):
        keep = np.argsort(z)[-top_k:]
        masked = np.full_like(z, -np.inf)
        masked[keep] = z[keep]
        z = masked
    e = np.exp(z - z.max())
    return e / e.sum()


def sample(logits, rng, temperature=1.0, top_k=None):
    """Finished for you: draw one token id."""
    p = sample_probs(logits, temperature, top_k)
    return int(rng.choice(len(p), p=p))
