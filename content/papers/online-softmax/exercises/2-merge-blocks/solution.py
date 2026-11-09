import math

import numpy as np


def block_stats(chunk):
    """Finished for you: the summary (m, d) of one block."""
    chunk = np.asarray(chunk, dtype=float)
    m = float(chunk.max())
    return m, float(np.exp(chunk - m).sum())


def combine(a, b):
    """Equation 4: merge two (m, d) summaries."""
    (m_a, d_a), (m_b, d_b) = a, b
    m = max(m_a, m_b)
    return m, d_a * math.exp(m_a - m) + d_b * math.exp(m_b - m)


def blocked_softmax(x, block_size):
    """Softmax of x, with (m, d) computed per block and merged with combine."""
    x = np.asarray(x, dtype=float)
    stats = [block_stats(x[i:i + block_size]) for i in range(0, len(x), block_size)]
    m, d = stats[0]
    for s in stats[1:]:
        m, d = combine((m, d), s)
    return np.exp(x - m) / d
