import numpy as np


def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def msa(x, p):
    """Finished for you: multi-head self-attention, no mask, p["h"] heads."""
    n, D = x.shape
    h = p["h"]
    split = lambda M: (x @ M).reshape(n, h, D // h).transpose(1, 0, 2)
    q, k, v = split(p["Wq"]), split(p["Wk"]), split(p["Wv"])
    w = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(D // h))
    return (w @ v).transpose(1, 0, 2).reshape(n, D) @ p["Wo"]


def mlp(x, p):
    """Finished for you: two linear layers with a GELU between them."""
    return gelu(x @ p["W1"] + p["c1"]) @ p["W2"] + p["c2"]


def encoder_block(z, p):
    """Equations 2 and 3 (pre-LN). z has shape (N + 1, D)."""
    raise NotImplementedError


def vit_readout(z0, blocks, g, b):
    """Equation 4: run the blocks, return LN of the class-token row, shape (D,)."""
    raise NotImplementedError
