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
