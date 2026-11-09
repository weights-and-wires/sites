import numpy as np


def layer_norm(x, eps=1e-6):
    """Finished for you: LayerNorm over the last axis, no learned scale or shift."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def modulate(h, shift, scale):
    """Adaptive scale and shift: h * (1 + scale) + shift."""
    return h * (1 + scale) + shift


def dit_block(x, c, W, b, attn, mlp):
    """One adaLN-Zero DiT block.

    x: (T, d) tokens. c: (d,) conditioning vector. W: (d, 6d). b: (6d,).
    attn and mlp are functions that map (T, d) to (T, d).
    """
    shift1, scale1, gate1, shift2, scale2, gate2 = np.split(c @ W + b, 6)
    x = x + gate1 * attn(modulate(layer_norm(x), shift1, scale1))
    x = x + gate2 * mlp(modulate(layer_norm(x), shift2, scale2))
    return x
