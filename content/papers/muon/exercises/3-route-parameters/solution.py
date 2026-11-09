import numpy as np

ADAMW_PREFIXES = ("tok_emb", "pos_emb", "wte", "wpe", "lm_head", "head", "classifier")


def route(name, shape):
    """Return "muon" or "adamw" for a parameter with this full name and shape."""
    if name.startswith(ADAMW_PREFIXES):
        return "adamw"
    if len(shape) in (2, 4):
        return "muon"
    return "adamw"


def flatten_for_muon(G):
    """Return the 2D matrix to orthogonalize: (out, in, kh, kw) becomes (out, in * kh * kw)."""
    if G.ndim == 2:
        return G
    return G.reshape(G.shape[0], -1)
