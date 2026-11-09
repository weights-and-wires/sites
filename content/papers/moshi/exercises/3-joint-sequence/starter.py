import numpy as np


def apply_delays(codes, delays):
    """Finished for you. Shift row q right by delays[q] steps, filling the front with 0."""
    codes = np.asarray(codes)
    Q, T = codes.shape
    out = np.zeros_like(codes)
    for q, d in enumerate(delays):
        out[q, d:] = codes[q, : T - d]
    return out


def joint_sequence(W, A, B, tau):
    """Return the (2Q + 1, T) array of Equation 6."""
    raise NotImplementedError


def sampled_rows(Q):
    """0-based rows that Moshi samples: its text row and its own audio rows."""
    raise NotImplementedError
