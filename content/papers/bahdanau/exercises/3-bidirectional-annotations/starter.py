import numpy as np


def annotations(X, Wf, Uf, Wb, Ub):
    """Return the bidirectional annotations h_j = [forward_j; backward_j].

    X: (Tx, m); Wf, Wb: (n, m); Uf, Ub: (n, n)
    Returns an array of shape (Tx, 2n).
    """
    raise NotImplementedError
