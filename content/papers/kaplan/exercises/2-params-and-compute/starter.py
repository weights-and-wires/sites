PF_DAY = 1e15 * 24 * 3600  # floating point operations in one PF-day


def non_embedding_params(n_layer, d_model, d_attn=None, d_ff=None):
    """N = 2 * d_model * n_layer * (2 * d_attn + d_ff), without embeddings or biases.

    d_attn defaults to d_model and d_ff defaults to 4 * d_model.
    """
    raise NotImplementedError


def embedding_params(n_vocab, n_ctx, d_model):
    """Token and position embedding parameters: (n_vocab + n_ctx) * d_model."""
    raise NotImplementedError


def forward_flops_per_token(n_layer, d_model, n_ctx, d_attn=None, d_ff=None):
    """C_forward = 2 N + 2 * n_layer * n_ctx * d_attn."""
    raise NotImplementedError


def training_compute_pf_days(n_params, batch_tokens, steps):
    """C = 6 * N * B * S floating point operations, reported in PF-days."""
    raise NotImplementedError
