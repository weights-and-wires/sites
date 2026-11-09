import numpy as np


def attainable(intensity, peak, bw):
    """Attainable GFLOP/s: min(peak, bw * intensity). Works on scalars and arrays."""
    return np.minimum(peak, bw * np.asarray(intensity, dtype=float))


def ridge_point(peak, bw):
    """Intensity (FLOPs/byte) at which the bandwidth roof reaches the compute roof."""
    return peak / bw


def is_memory_bound(intensity, peak, bw):
    """True when the kernel sits on the slanted part of the roof."""
    return bw * intensity < peak
