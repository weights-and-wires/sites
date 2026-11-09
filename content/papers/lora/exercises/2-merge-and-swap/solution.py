import numpy as np


def lora_forward(X, W0, A, B, alpha):
    """Finished for you: X W0^T + (alpha / r) X A^T B^T."""
    r = A.shape[0]
    return X @ W0.T + (alpha / r) * (X @ A.T) @ B.T


def merge(W0, A, B, alpha):
    """Return W0 + (alpha / r) B A, shape (d, k)."""
    r = A.shape[0]
    return W0 + (alpha / r) * (B @ A)


def unmerge(W, A, B, alpha):
    """Return W - (alpha / r) B A: the weight before this adapter was merged."""
    r = A.shape[0]
    return W - (alpha / r) * (B @ A)


def swap(W, old, new, alpha):
    """old and new are (A, B) pairs. Remove old from W, then merge new."""
    W0 = unmerge(W, old[0], old[1], alpha)
    return merge(W0, new[0], new[1], alpha)
