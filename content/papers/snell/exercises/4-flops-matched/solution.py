def pretrain_flops(n_params, d_tokens):
    """Pretraining FLOPs, 6 * N * D."""
    return 6 * n_params * d_tokens


def inference_flops(n_params, d_tokens):
    """Inference FLOPs, 2 * N * D."""
    return 2 * n_params * d_tokens


def matched_multiplier(m, r):
    """Factor on the small model's inference compute that matches a model m times larger.

    r is D_inference / D_pretrain.
    """
    return m + 3 * (m - 1) / r
