import math

import numpy as np


def block_stats(chunk):
    """Finished for you: the summary (m, d) of one block."""
    chunk = np.asarray(chunk, dtype=float)
    m = float(chunk.max())
    return m, float(np.exp(chunk - m).sum())


def combine(a, b):
    """Equation 4: merge two (m, d) summaries."""
    raise NotImplementedError


def blocked_softmax(x, block_size):
    """Softmax of x, with (m, d) computed per block and merged with combine."""
    raise NotImplementedError
