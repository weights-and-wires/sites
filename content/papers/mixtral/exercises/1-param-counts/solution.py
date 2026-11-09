def count_params(cfg):
    """Return {"total": int, "active": int} for a Mixtral-style config (Table 1)."""
    d = cfg["dim"]
    attn = 2 * d * cfg["n_heads"] * cfg["head_dim"] + 2 * d * cfg["n_kv_heads"] * cfg["head_dim"]
    expert = 3 * d * cfg["hidden_dim"]
    router = d * cfg["num_experts"] if cfg["num_experts"] > 1 else 0
    norms = 2 * d
    fixed = 2 * cfg["vocab_size"] * d + d
    layer_total = attn + cfg["num_experts"] * expert + router + norms
    layer_active = attn + cfg["top_k_experts"] * expert + router + norms
    return {
        "total": cfg["n_layers"] * layer_total + fixed,
        "active": cfg["n_layers"] * layer_active + fixed,
    }
