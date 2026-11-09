import numpy as np


def to_bfloat16(x):
    """Round an FP32 array to the nearest bfloat16 value, ties to even.

    Returns a float32 array.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + bias) & np.uint32(0xFFFF0000)).view(np.float32)
