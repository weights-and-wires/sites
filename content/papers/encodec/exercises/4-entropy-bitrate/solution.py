import math

import numpy as np


def _steps(fs, strides):
    return fs / math.prod(strides)


def tokens_per_second(fs, strides, n_q):
    """Tokens per second: (fs / prod(strides)) * n_q."""
    return _steps(fs, strides) * n_q


def bitrate_kbps(fs, strides, n_q, n_codes=1024):
    """Fixed-length bitrate in kbps."""
    return tokens_per_second(fs, strides, n_q) * math.log2(n_codes) / 1000


def n_q_for_kbps(kbps, fs, strides, n_codes=1024):
    """Number of codebooks that gives exactly kbps. Raise ValueError if there is none."""
    per = bitrate_kbps(fs, strides, 1, n_codes)
    n = kbps / per
    r = round(n)
    if r < 1 or abs(n - r) > 1e-6:
        raise ValueError(f"{kbps} kbps is not a whole number of codebooks ({per} kbps each)")
    return int(r)


def entropy_kbps(probs, fs, strides):
    """Ideal entropy-coded kbps: sum over codebooks of the entropy in bits, times steps per second."""
    probs = np.asarray(probs, dtype=float)
    bits = 0.0
    for p in probs:
        p = p[p > 0]
        bits += -(p * np.log2(p)).sum()
    return float(bits * _steps(fs, strides) / 1000)
