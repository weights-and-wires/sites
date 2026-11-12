import numpy as np


def masked_lm_loss(logits, labels, ignore_index=-100):
    """Mean cross-entropy over positions where labels != ignore_index."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels)
    sel = labels != ignore_index
    if not sel.any():
        return 0.0
    z = logits[sel]
    z = z - z.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(z.shape[0]), labels[sel]]
    return float(nll.mean())
