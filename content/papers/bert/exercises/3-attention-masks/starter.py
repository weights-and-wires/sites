import numpy as np


def attention(x, Wq, Wk, Wv, causal=False):
    """Single-head scaled dot-product self-attention.

    Returns (out, weights). With causal=True, position i reads only j <= i.
    """
    raise NotImplementedError
