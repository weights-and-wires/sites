import numpy as np


def paged_attention(q, pool_K, pool_V, block_table, n_tokens):
    """Attention of one query over a KV cache stored in non-contiguous blocks (Equation 4).

    q: (d,). pool_K, pool_V: (num_blocks, B, d). block_table: physical block IDs
    in logical order. n_tokens: number of filled slots in the sequence.
    """
    raise NotImplementedError
