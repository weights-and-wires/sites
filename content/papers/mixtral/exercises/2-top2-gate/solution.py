import numpy as np


def top_k_gate(logits, k=2):
    """Return (idx, weights), both of shape (T, k); weights = Softmax(TopK(logits))."""
    idx = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, idx, axis=-1)
    e = np.exp(top - top.max(axis=-1, keepdims=True))
    return idx, e / e.sum(axis=-1, keepdims=True)
