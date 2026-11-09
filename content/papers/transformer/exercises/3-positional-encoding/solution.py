import numpy as np


def sinusoidal_pe(n, d):
    """Return an (n, d) array of sinusoidal positional encodings; d is even.

    pe[pos, 2i]   = sin(pos / 10000 ** (2i / d))
    pe[pos, 2i+1] = cos(pos / 10000 ** (2i / d))
    """
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    angle = pos / 10000 ** (2 * i / d)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(angle)
    pe[:, 1::2] = np.cos(angle)
    return pe
