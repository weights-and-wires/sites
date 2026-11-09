import numpy as np


def scale_roundtrip(g, scale):
    """Scale g in FP32, store it in FP16, and return the unscaled FP32 array."""
    s = np.float32(scale)
    stored = (np.asarray(g, dtype=np.float32) * s).astype(np.float16)
    return stored.astype(np.float32) / s


def choose_scale(max_abs_grad, fp16_max=65504.0):
    """Return the largest power of two S with S * max_abs_grad <= fp16_max."""
    return float(2.0 ** np.floor(np.log2(fp16_max / max_abs_grad)))
