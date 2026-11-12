def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value=2):
    """Bytes of KV cache for one token: keys and values, in every layer."""
    raise NotImplementedError


def transfer_seconds(n_bytes, bandwidth_gbit):
    """Seconds to move n_bytes over a link of bandwidth_gbit gigabits per second."""
    raise NotImplementedError


def min_bandwidth_gbit(n_bytes, rate):
    """Gigabits per second needed to carry `rate` requests per second of n_bytes each."""
    raise NotImplementedError
