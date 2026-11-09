def block_params(d):
    """Parameters of one block: attention, feed-forward, and two layer norms."""
    raise NotImplementedError


def n_params(layers, d_model, n_vocab=50257, n_ctx=2048):
    """Total parameters with tied input and output embeddings and a final layer norm."""
    raise NotImplementedError


def train_flops(n_params, n_tokens):
    """Appendix D: 6 flops per parameter per token, ignoring attention."""
    raise NotImplementedError


def petaflop_days(flops):
    """Convert flops to petaflop/s-days."""
    raise NotImplementedError


def attention_fraction(d_model, n_ctx):
    """Share of forward flops in QK^T and AV at full context, dense attention."""
    raise NotImplementedError
