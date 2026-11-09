import math

import numpy as np


def expert_capacity(T, N, capacity_factor):
    """Equation 3, rounded up to an integer."""
    raise NotImplementedError


def keep_mask(idx, N, capacity):
    """idx: (T,) chosen expert per token. Returns a (T,) boolean array.

    Token t is kept when fewer than `capacity` earlier tokens chose idx[t].
    """
    raise NotImplementedError


def switch_layer(x, idx, gate, keep, W):
    """x: (T, d), idx: (T,), gate: (T,), keep: (T,) bool, W: (N, d, d).

    Kept tokens:    x + gate * (x @ W[idx])
    Dropped tokens: x (the residual connection carries them past the layer)
    """
    raise NotImplementedError
