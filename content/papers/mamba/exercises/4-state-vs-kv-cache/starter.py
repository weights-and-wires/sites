def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, bytes_per_elem=2):
    """Bytes of the key and value cache of a Transformer after seq_len tokens, one sequence."""
    raise NotImplementedError


def mamba_state_bytes(n_layers, d_model, expand=2, d_state=16, d_conv=4, bytes_per_elem=2):
    """Bytes of the Mamba recurrent state (SSM state plus convolution buffer), one sequence."""
    raise NotImplementedError


def crossover_length(n_layers, n_heads, head_dim, state_bytes, bytes_per_elem=2):
    """Smallest seq_len for which the KV cache is strictly larger than state_bytes."""
    raise NotImplementedError
