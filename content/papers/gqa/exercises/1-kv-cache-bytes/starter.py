def kv_cache_bytes_per_token(n_layers, n_kv_heads, d_head, bytes_per_value=2):
    """Bytes added to the KV cache by one token.

    Every layer stores one key vector and one value vector of length d_head
    for each of the n_kv_heads key-value heads.
    """
    raise NotImplementedError


def max_concurrent_sequences(gpu_bytes, weight_bytes, seq_len, n_layers, n_kv_heads, d_head, bytes_per_value=2):
    """How many full-length sequences fit in the memory left after the weights.

    Returns an int, rounded down.
    """
    raise NotImplementedError
