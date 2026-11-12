import numpy as np

E4M3_MAX = 448.0


def round_e4m3(x):
    """Round to the nearest E4M3 value (ties to even) and saturate at 448."""
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    e = np.maximum(np.floor(np.log2(np.where(a > 0, a, 1.0))), -6)
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.round(a / step) * step, E4M3_MAX)


def quantize_blocks(x, block_rows):
    """Quantize and dequantize `x` with one amax / 448 scale per block of rows."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for i in range(0, x.shape[0], block_rows):
        blk = x[i:i + block_rows]
        amax = np.abs(blk).max()
        if amax == 0:
            continue
        scale = amax / E4M3_MAX
        out[i:i + block_rows] = round_e4m3(blk / scale) * scale
    return out


def hadamard_transform(x):
    """Orthonormal Walsh-Hadamard transform over the last axis (length a power of two)."""
    x = np.array(x, dtype=float)
    d = x.shape[-1]
    h = 1
    while h < d:
        x = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        x = np.stack([x[..., 0, :] + x[..., 1, :], x[..., 0, :] - x[..., 1, :]], axis=-2)
        x = x.reshape(*x.shape[:-3], d)
        h *= 2
    return x / np.sqrt(d)


def incoherent(x, signs):
    """Multiply by a random +-1 diagonal matrix, then by a Hadamard matrix."""
    return hadamard_transform(x * signs)


def relative_score_error(Q, K, block_rows, signs=None):
    """Relative RMSE of Q K^T after optional rotation and block quantization."""
    S = Q @ K.T
    if signs is not None:
        Q, K = incoherent(Q, signs), incoherent(K, signs)
    S_hat = quantize_blocks(Q, block_rows) @ quantize_blocks(K, block_rows).T
    return np.sqrt(np.mean((S - S_hat) ** 2)) / np.sqrt(np.mean(S ** 2))
