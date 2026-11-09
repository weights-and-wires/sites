import numpy as np


def flash_attention2_forward(Q, K, V, Br, Bc, tau=1.0):
    """Algorithm 1 of FlashAttention-2: outer loop over Q blocks.

    Q: (N, d), K: (M, d), V: (M, dv). Returns (O, L) with O of shape
    (N, dv) and the logsumexp L of shape (N,).
    """
    raise NotImplementedError
