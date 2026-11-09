import numpy as np


def mean_pool_heads(W, H, G):
    """Mean-pool the H head blocks of W into G group blocks.

    W: (d_model, H * d_k), where column block i of width d_k is head i.
    Group g averages heads g * (H // G) through (g + 1) * (H // G) - 1.
    Returns (d_model, G * d_k). G divides H.
    """
    raise NotImplementedError
