def pretrain_flops(n_params, d_tokens):
    """Pretraining FLOPs, 6 * N * D."""
    raise NotImplementedError


def inference_flops(n_params, d_tokens):
    """Inference FLOPs, 2 * N * D."""
    raise NotImplementedError


def matched_multiplier(m, r):
    """Factor on the small model's inference compute that matches a model m times larger.

    r is D_inference / D_pretrain.
    """
    raise NotImplementedError
