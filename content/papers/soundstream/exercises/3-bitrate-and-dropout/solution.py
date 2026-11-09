import math

import numpy as np


def frame_rate(fs, strides):
    """Embedding frames per second: fs divided by the product of the strides."""
    return fs / math.prod(strides)


def bitrate(fs, strides, n_q, n_codes):
    """Bits per second when n_q quantizers with n_codes entries encode each frame."""
    return frame_rate(fs, strides) * n_q * math.log2(n_codes)


def num_quantizers(target_bps, fs, strides, n_codes):
    """Largest integer n_q with bitrate(fs, strides, n_q, n_codes) <= target_bps."""
    per_quantizer = bitrate(fs, strides, 1, n_codes)
    return int(math.floor(target_bps / per_quantizer + 1e-9))


def sample_nq(n_q_max, rng):
    """Quantizer dropout: an integer drawn uniformly from 1..n_q_max, inclusive."""
    return int(rng.integers(1, n_q_max + 1))
