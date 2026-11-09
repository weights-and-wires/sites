import numpy as np


def flash_attention_forward(Q, K, V, Br, Bc, tau=1.0):
    """Algorithm 1: tiled attention with an online softmax.

    Q: (N, d), K: (M, d), V: (M, dv). Returns (O, m, l) with O of shape
    (N, dv), and m and l of shape (N,).
    """
    raise NotImplementedError
