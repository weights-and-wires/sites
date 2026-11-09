def kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, bytes_per_elem=2):
    """Bytes of the key and value cache of a Transformer after seq_len tokens, one sequence."""
    return 2 * n_layers * n_heads * head_dim * seq_len * bytes_per_elem


def mamba_state_bytes(n_layers, d_model, expand=2, d_state=16, d_conv=4, bytes_per_elem=2):
    """Bytes of the Mamba recurrent state (SSM state plus convolution buffer), one sequence."""
    d_inner = expand * d_model
    return n_layers * (d_inner * d_state + d_inner * (d_conv - 1)) * bytes_per_elem


def crossover_length(n_layers, n_heads, head_dim, state_bytes, bytes_per_elem=2):
    """Smallest seq_len for which the KV cache is strictly larger than state_bytes."""
    per_token = kv_cache_bytes(n_layers, n_heads, head_dim, 1, bytes_per_elem)
    return state_bytes // per_token + 1
