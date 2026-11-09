import numpy as np


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, g, b, eps=1e-5):
    """Finished for you. Normalizes over the last axis."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def attention(x, p, n_heads):
    """Finished for you: causal multi-head self-attention on x of shape (T, d)."""
    T, d = x.shape
    dk = d // n_heads
    qkv = x @ p["wqkv"] + p["bqkv"]
    q, k, v = np.split(qkv, 3, axis=-1)
    split = lambda a: a.reshape(T, n_heads, dk).transpose(1, 0, 2)
    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(dk)
    scores = np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -1e9)
    out = (softmax(scores) @ v).transpose(1, 0, 2).reshape(T, d)
    return out @ p["wo"] + p["bo"]


def ffn(x, p):
    """Finished for you: two linear maps with a GELU between them."""
    return gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def gpt_block(x, p, n_heads):
    """One pre-LN block: x + attention(LN1(x)), then x + ffn(LN2(x))."""
    x = x + attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, n_heads)
    x = x + ffn(layer_norm(x, p["ln2_g"], p["ln2_b"]), p)
    return x


def gpt_forward(tokens, params, n_heads):
    """Token ids of shape (T,) to logits of shape (T, V), with tied embeddings."""
    tokens = np.asarray(tokens)
    x = params["wte"][tokens] + params["wpe"][: len(tokens)]
    for p in params["blocks"]:
        x = gpt_block(x, p, n_heads)
    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["wte"].T


def residual_scale(n_blocks):
    """The factor 1 / sqrt(N), where N counts the residual layers."""
    return 1.0 / np.sqrt(2 * n_blocks)
