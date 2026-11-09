def block_params(d):
    """Parameters of one block: attention, feed-forward, and two layer norms."""
    attn = d * 3 * d + 3 * d + d * d + d
    ffn = d * 4 * d + 4 * d + 4 * d * d + d
    ln = 2 * (2 * d)
    return attn + ffn + ln


def n_params(layers, d_model, n_vocab=50257, n_ctx=2048):
    """Total parameters with tied input and output embeddings and a final layer norm."""
    return n_vocab * d_model + n_ctx * d_model + layers * block_params(d_model) + 2 * d_model


def train_flops(n_params, n_tokens):
    """Appendix D: 6 flops per parameter per token, ignoring attention."""
    return 6 * n_params * n_tokens


def petaflop_days(flops):
    """Convert flops to petaflop/s-days."""
    return flops / 8.64e19


def attention_fraction(d_model, n_ctx):
    """Share of forward flops in QK^T and AV at full context, dense attention."""
    attn = 4 * n_ctx * d_model
    weights = 24 * d_model**2
    return attn / (attn + weights)
