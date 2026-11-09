import math

import numpy as np


def tokens_per_second(fs, strides, n_q):
    """Tokens per second: (fs / prod(strides)) * n_q."""
    raise NotImplementedError


def bitrate_kbps(fs, strides, n_q, n_codes=1024):
    """Fixed-length bitrate in kbps."""
    raise NotImplementedError


def n_q_for_kbps(kbps, fs, strides, n_codes=1024):
    """Number of codebooks that gives exactly kbps. Raise ValueError if there is none."""
    raise NotImplementedError


def entropy_kbps(probs, fs, strides):
    """Ideal entropy-coded kbps: sum over codebooks of the entropy in bits, times steps per second."""
    raise NotImplementedError
