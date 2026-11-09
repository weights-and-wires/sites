import math

import numpy as np


def expert_capacity(T, N, capacity_factor):
    """Equation 3, rounded up to an integer."""
    return int(math.ceil(T / N * capacity_factor))


def keep_mask(idx, N, capacity):
    """idx: (T,) chosen expert per token. Returns a (T,) boolean array.

    Token t is kept when fewer than `capacity` earlier tokens chose idx[t].
    """
    seen = np.zeros(N, dtype=int)
    keep = np.zeros(len(idx), dtype=bool)
    for t, e in enumerate(idx):
        keep[t] = seen[e] < capacity
        seen[e] += 1
    return keep


def switch_layer(x, idx, gate, keep, W):
    """x: (T, d), idx: (T,), gate: (T,), keep: (T,) bool, W: (N, d, d).

    Kept tokens:    x + gate * (x @ W[idx])
    Dropped tokens: x (the residual connection carries them past the layer)
    """
    expert_out = np.einsum("td,tde->te", x, W[idx])
    update = gate[:, None] * expert_out * keep[:, None]
    return x + update
