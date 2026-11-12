import numpy as np


def _unit(x):
    x = np.asarray(x, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def info_nce(q, d_pos, d_hard=None, tau=0.02):
    """Mean InfoNCE loss over the batch with in-batch and hard negatives.

    q, d_pos: (B, d). d_hard: None or (B, d). Returns a float.
    """
    q = _unit(q)
    cand = _unit(d_pos)
    if d_hard is not None:
        cand = np.concatenate([cand, _unit(d_hard)], axis=0)
    scores = (q @ cand.T) / tau
    m = scores.max(axis=1, keepdims=True)
    log_denom = m[:, 0] + np.log(np.exp(scores - m).sum(axis=1))
    pos = scores[np.arange(len(q)), np.arange(len(q))]
    return float(np.mean(log_denom - pos))
