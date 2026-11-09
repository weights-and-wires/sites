import numpy as np


def guided_output(out_cond, out_uncond, s, C=4, n_guided=3):
    """Apply classifier-free guidance to the first n_guided noise channels.

    out_cond, out_uncond: (2*C, H, W). Channels 0..C-1 are the noise prediction,
    channels C..2C-1 are the covariance. Returns an array of the same shape.
    """
    out = out_cond.copy()
    out[:n_guided] = out_uncond[:n_guided] + s * (out_cond[:n_guided] - out_uncond[:n_guided])
    return out
