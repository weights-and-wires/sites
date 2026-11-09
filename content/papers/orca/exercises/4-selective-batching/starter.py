import numpy as np


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def causal_attention(q, k, v):
    """Finished for you. Single head, one request: q, k, v have shape (L, H)."""
    L, H = q.shape
    scores = q @ k.T / np.sqrt(H)
    scores = np.where(np.tril(np.ones((L, L), dtype=bool)), scores, -1e9)
    return softmax(scores) @ v


def selective_batch_layer(x, lens, Wqkv, Wo):
    """x: (sum(lens), H). Return an array of shape (sum(lens), H)."""
    raise NotImplementedError
