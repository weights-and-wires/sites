import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def rmsnorm(x, g, eps=1e-6):
    """Finished for you. Normalizes the last axis."""
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * g


def silu(a):
    return a / (1.0 + np.exp(-a))


def swiglu_ffn(x, W1, V, W2):
    """Finished for you. (SiLU(x W1) * (x V)) W2."""
    return (silu(x @ W1) * (x @ V)) @ W2


def apply_rope(x, positions, base=10000.0):
    """Finished for you. x: (n, d_head); rotates adjacent pairs by position angles."""
    n, d = x.shape
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    angles = np.asarray(positions, dtype=float)[:, None] * theta[None, :]
    c, s = np.cos(angles), np.sin(angles)
    out = np.empty_like(x, dtype=float)
    out[:, 0::2] = x[:, 0::2] * c - x[:, 1::2] * s
    out[:, 1::2] = x[:, 0::2] * s + x[:, 1::2] * c
    return out


def llama_block(x, p, n_heads, positions=None):
    """One LLaMA decoder block.

    x: (n, d). p: dict with g1, g2, Wq, Wk, Wv, Wo, W1, V, W2 (see the goal).
    Returns (n, d): x + attention(rmsnorm(x)), then + swiglu(rmsnorm(.)).
    """
    raise NotImplementedError
