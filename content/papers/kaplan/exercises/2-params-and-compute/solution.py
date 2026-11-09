PF_DAY = 1e15 * 24 * 3600  # floating point operations in one PF-day


def non_embedding_params(n_layer, d_model, d_attn=None, d_ff=None):
    """N = 2 * d_model * n_layer * (2 * d_attn + d_ff), without embeddings or biases.

    d_attn defaults to d_model and d_ff defaults to 4 * d_model.
    """
    d_attn = d_model if d_attn is None else d_attn
    d_ff = 4 * d_model if d_ff is None else d_ff
    return 2 * d_model * n_layer * (2 * d_attn + d_ff)


def embedding_params(n_vocab, n_ctx, d_model):
    """Token and position embedding parameters: (n_vocab + n_ctx) * d_model."""
    return (n_vocab + n_ctx) * d_model


def forward_flops_per_token(n_layer, d_model, n_ctx, d_attn=None, d_ff=None):
    """C_forward = 2 N + 2 * n_layer * n_ctx * d_attn."""
    d_attn = d_model if d_attn is None else d_attn
    n = non_embedding_params(n_layer, d_model, d_attn, d_ff)
    return 2 * n + 2 * n_layer * n_ctx * d_attn


def training_compute_pf_days(n_params, batch_tokens, steps):
    """C = 6 * N * B * S floating point operations, reported in PF-days."""
    return 6 * n_params * batch_tokens * steps / PF_DAY
