def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value):
    """Bytes that one token adds to the KV cache."""
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def cache_slots(n, sinks, window):
    """Tokens held by a sink-plus-window cache after n tokens."""
    return min(n, sinks + window)


def cache_bytes(n, per_token, sinks=None, window=None):
    """Cache size in bytes. A full cache when sinks is None."""
    if sinks is None:
        return n * per_token
    return cache_slots(n, sinks, window) * per_token
