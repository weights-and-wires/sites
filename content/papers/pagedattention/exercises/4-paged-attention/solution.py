import numpy as np


def paged_attention(q, pool_K, pool_V, block_table, n_tokens):
    """Attention of one query over a KV cache stored in non-contiguous blocks (Equation 4)."""
    d = q.shape[-1]
    B = pool_K.shape[1]
    scores = []
    values = []
    for j, phys in enumerate(block_table):
        filled = min(B, n_tokens - j * B)
        if filled <= 0:
            break
        scores.append(pool_K[phys, :filled] @ q / np.sqrt(d))
        values.append(pool_V[phys, :filled])
    s = np.concatenate(scores)
    e = np.exp(s - s.max())
    a = e / e.sum()
    return a @ np.concatenate(values)
