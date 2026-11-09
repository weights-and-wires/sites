import numpy as np

ADAMW_PREFIXES = ("tok_emb", "pos_emb", "wte", "wpe", "lm_head", "head", "classifier")


def route(name, shape):
    """Return "muon" or "adamw" for a parameter with this full name and shape."""
    raise NotImplementedError


def flatten_for_muon(G):
    """Return the 2D matrix to orthogonalize: (out, in, kh, kw) becomes (out, in * kh * kw)."""
    raise NotImplementedError
