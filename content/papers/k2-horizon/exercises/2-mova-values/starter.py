import numpy as np


def silu(z):
    """z * sigmoid(z)."""
    raise NotImplementedError


def route(x, W_r, k):
    """Top-k experts per token (best first) and the softmax of their logits."""
    raise NotImplementedError


def mova_values(x, W_r, W_v, k):
    """Equation 6: sum over chosen experts of p_i * SiLU(x W_v[i]). Shape (T, d_v)."""
    raise NotImplementedError
