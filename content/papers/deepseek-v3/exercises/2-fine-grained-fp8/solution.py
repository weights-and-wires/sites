import numpy as np

E4M3_MAX, MBITS, BIAS = 448.0, 3, 7


def quantize(x):
    """Finished for you: round to the nearest E4M3 value, saturating on overflow."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    _, p = np.frexp(np.where(a > 0, a, 1.0))
    e = np.maximum(p - 1, 1 - BIAS)
    spacing = 2.0 ** (e - MBITS)
    return np.sign(x) * np.minimum(np.round(a / spacing) * spacing, E4M3_MAX)


def per_tensor_roundtrip(x):
    """Finished for you: one scale for the whole tensor."""
    x = np.asarray(x, dtype=np.float64)
    s = E4M3_MAX / np.abs(x).max()
    return quantize(x * s) / s


def _scale_from_amax(amax):
    amax = np.where(amax == 0, E4M3_MAX, amax)
    return E4M3_MAX / amax


def tile_scales(x, n_c=128):
    """Scale of each 1 x n_c tile: 448 / amax, or 1.0 for an all-zero tile. Shape (R, C // n_c)."""
    x = np.asarray(x, dtype=np.float64)
    r, c = x.shape
    g = x.reshape(r, c // n_c, n_c)
    return _scale_from_amax(np.abs(g).max(axis=2))


def tile_roundtrip(x, n_c=128):
    """Scale each 1 x n_c tile, cast to E4M3, unscale. Same shape as x."""
    x = np.asarray(x, dtype=np.float64)
    r, c = x.shape
    g = x.reshape(r, c // n_c, n_c)
    s = _scale_from_amax(np.abs(g).max(axis=2, keepdims=True))
    return (quantize(g * s) / s).reshape(r, c)


def block_roundtrip(x, n_c=128):
    """Scale each n_c x n_c block, cast to E4M3, unscale. Same shape as x."""
    x = np.asarray(x, dtype=np.float64)
    r, c = x.shape
    g = x.reshape(r // n_c, n_c, c // n_c, n_c)
    s = _scale_from_amax(np.abs(g).max(axis=(1, 3), keepdims=True))
    return (quantize(g * s) / s).reshape(r, c)
