import numpy as np


def rotate(x, positions, base=10000.0):
    """Rotary transform. x: (n, d) with d even, positions: (n,). Pair j rotates by position * base ** (-2j / d)."""
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    angle = np.asarray(positions, dtype=float)[:, None] * theta[None, :]
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin
    out[..., 1::2] = a * sin + b * cos
    return out


def cache_positions(kept):
    """Positions of the cache rows: 0, 1, ..., len(kept) - 1."""
    return np.arange(len(kept))


def streaming_scores(q, K, kept, base=10000.0):
    """Scores of the current token's query against every cached key, using cache positions.

    q: (d,) un-rotated query of the current token.
    K: (m, d) un-rotated keys. The last row is the current token's key.
    kept: original token index of each row of K (ignored for positions).
    """
    positions = cache_positions(kept)
    q_rot = rotate(np.asarray(q, dtype=float)[None, :], positions[-1:], base)[0]
    K_rot = rotate(K, positions, base)
    return K_rot @ q_rot
