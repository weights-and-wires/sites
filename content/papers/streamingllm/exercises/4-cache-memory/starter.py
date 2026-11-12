def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value):
    """Bytes that one token adds to the KV cache."""
    raise NotImplementedError


def cache_slots(n, sinks, window):
    """Tokens held by a sink-plus-window cache after n tokens."""
    raise NotImplementedError


def cache_bytes(n, per_token, sinks=None, window=None):
    """Cache size in bytes. A full cache when sinks is None."""
    raise NotImplementedError
