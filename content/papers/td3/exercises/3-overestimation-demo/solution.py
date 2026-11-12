import numpy as np


def bias_estimates(n_actions, sigma, trials, seed):
    """Return {"max": ..., "double": ..., "clipped": ...}, each a float.

    All true action values are 0, so each number is a bias.
    """
    rng = np.random.default_rng(seed)
    qa = rng.normal(0.0, sigma, (trials, n_actions))
    qb = rng.normal(0.0, sigma, (trials, n_actions))
    star = qa.argmax(axis=1)
    rows = np.arange(trials)
    a_at = qa[rows, star]
    b_at = qb[rows, star]
    return {
        "max": float(qa.max(axis=1).mean()),
        "double": float(b_at.mean()),
        "clipped": float(np.minimum(a_at, b_at).mean()),
    }
