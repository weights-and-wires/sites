import math

import numpy as np


def frame_rate(fs, strides):
    """Embedding frames per second: fs divided by the product of the strides."""
    raise NotImplementedError


def bitrate(fs, strides, n_q, n_codes):
    """Bits per second when n_q quantizers with n_codes entries encode each frame."""
    raise NotImplementedError


def num_quantizers(target_bps, fs, strides, n_codes):
    """Largest integer n_q with bitrate(fs, strides, n_q, n_codes) <= target_bps."""
    raise NotImplementedError


def sample_nq(n_q_max, rng):
    """Quantizer dropout: an integer drawn uniformly from 1..n_q_max, inclusive."""
    raise NotImplementedError
