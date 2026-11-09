import numpy as np


def top_k_gate(logits, k=2):
    """Return (idx, weights), both of shape (T, k); weights = Softmax(TopK(logits))."""
    raise NotImplementedError
