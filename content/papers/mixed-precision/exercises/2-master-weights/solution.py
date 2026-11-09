import numpy as np


def fp16_only_step(w16, g16, lr):
    """Naive SGD step: everything in FP16. Returns an FP16 array."""
    w16 = np.asarray(w16, dtype=np.float16)
    g16 = np.asarray(g16, dtype=np.float16)
    return (w16 - np.float16(lr) * g16).astype(np.float16)


def master_step(master, g16, lr):
    """Mixed precision SGD step. Returns (new_master FP32, new_w16 FP16)."""
    update = np.float32(lr) * np.asarray(g16, dtype=np.float16).astype(np.float32)
    new_master = (np.asarray(master, dtype=np.float32) - update).astype(np.float32)
    return new_master, new_master.astype(np.float16)
