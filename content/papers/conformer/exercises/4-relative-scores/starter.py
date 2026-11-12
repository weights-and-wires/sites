import numpy as np


def rel_scores(q, k, p, u, v):
    """Relative-position attention scores for one head.

    q, k: (T, dk). p: (2T - 1, dk), row m is the encoding of offset
    r = T - 1 - m. u, v: (dk,). Return a (T, T) array A with
    A[i, j] = ((q[i] + u) . k[j] + (q[i] + v) . p_{i-j}) / sqrt(dk).
    """
    raise NotImplementedError
