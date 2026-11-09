import numpy as np


def apply_delays(codes, delays):
    """Shift row q right by delays[q] steps, filling the front with 0."""
    codes = np.asarray(codes)
    Q, T = codes.shape
    out = np.zeros_like(codes)
    for q, d in enumerate(delays):
        out[q, d:] = codes[q, : T - d]
    return out


def undo_delays(V, delays):
    """Invert apply_delays on the steps that every row has."""
    V = np.asarray(V)
    Q, T = V.shape
    width = T - max(delays)
    return np.stack([V[q, d : d + width] for q, d in enumerate(delays)])


def latency_ms(delays, frame_ms=80):
    """Theoretical latency of a delay pattern, in milliseconds."""
    return (max(delays) + 1) * frame_ms
