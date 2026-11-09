import numpy as np


def scale_roundtrip(g, scale):
    """Scale g in FP32, store it in FP16, and return the unscaled FP32 array."""
    raise NotImplementedError


def choose_scale(max_abs_grad, fp16_max=65504.0):
    """Return the largest power of two S with S * max_abs_grad <= fp16_max."""
    raise NotImplementedError
