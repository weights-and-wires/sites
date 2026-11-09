import numpy as np


def attainable(intensity, peak, bw):
    """Attainable GFLOP/s: min(peak, bw * intensity). Works on scalars and arrays."""
    raise NotImplementedError


def ridge_point(peak, bw):
    """Intensity (FLOPs/byte) at which the bandwidth roof reaches the compute roof."""
    raise NotImplementedError


def is_memory_bound(intensity, peak, bw):
    """True when the kernel sits on the slanted part of the roof."""
    raise NotImplementedError
