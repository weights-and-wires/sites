import numpy as np


def apply_delays(codes, delays):
    """Shift row q right by delays[q] steps, filling the front with 0."""
    raise NotImplementedError


def undo_delays(V, delays):
    """Invert apply_delays on the steps that every row has."""
    raise NotImplementedError


def latency_ms(delays, frame_ms=80):
    """Theoretical latency of a delay pattern, in milliseconds."""
    raise NotImplementedError
