import numpy as np


def robust_z(y, w):
    """Equation 12 over a trailing window of w steps (not including t). Zeros before index w."""
    y = np.asarray(y, dtype=float)
    z = np.zeros_like(y)
    for t in range(w, len(y)):
        win = y[t - w:t]
        m = np.median(win)
        mad = np.median(np.abs(win - m))
        z[t] = (y[t] - m) / (1.4826 * mad)
    return z


def spike_episodes(z, on=5.0, off=2.0, calm=3):
    """(start, end) pairs: start at z > on, end at the third consecutive z < off."""
    out, start, run = [], None, 0
    for t, v in enumerate(z):
        if start is None:
            if v > on:
                start, run = t, 0
        else:
            run = run + 1 if v < off else 0
            if run == calm:
                out.append((start, t))
                start = None
    if start is not None:
        out.append((start, len(z) - 1))
    return out
