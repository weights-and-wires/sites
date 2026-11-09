import numpy as np


def sinusoidal_pe(n, d):
    """Return an (n, d) array of sinusoidal positional encodings; d is even.

    pe[pos, 2i]   = sin(pos / 10000 ** (2i / d))
    pe[pos, 2i+1] = cos(pos / 10000 ** (2i / d))
    """
    raise NotImplementedError
