import numpy as np


def robust_z(y, w):
    """Equation 12 over a trailing window of w steps (not including t). Zeros before index w."""
    raise NotImplementedError


def spike_episodes(z, on=5.0, off=2.0, calm=3):
    """(start, end) pairs: start at z > on, end at the third consecutive z < off."""
    raise NotImplementedError
