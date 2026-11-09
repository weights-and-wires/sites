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
    q, k, v = np.split(x @ Wqkv, 3, axis=1)
    merged = np.empty_like(x)
    start = 0
    for n in lens:
        s = slice(start, start + n)
        merged[s] = causal_attention(q[s], k[s], v[s])
        start += n
    return merged @ Wo
