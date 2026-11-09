import numpy as np


def fp16_only_step(w16, g16, lr):
    """Naive SGD step: everything in FP16. Returns an FP16 array."""
    raise NotImplementedError


def master_step(master, g16, lr):
    """Mixed precision SGD step. Returns (new_master FP32, new_w16 FP16)."""
    raise NotImplementedError
