import math

import numpy as np


def silu(a):
    """SiLU, also called swish with beta = 1: a * sigmoid(a). Elementwise."""
    raise NotImplementedError


def swiglu_ffn(x, W1, V, W2):
    """(SiLU(x W1) * (x V)) W2. x: (..., d); W1, V: (d, h); W2: (h, d)."""
    raise NotImplementedError


def ffn_hidden(d, multiple_of=256):
    """Hidden width: int(2 * 4 * d / 3) rounded up to a multiple of multiple_of."""
    raise NotImplementedError
