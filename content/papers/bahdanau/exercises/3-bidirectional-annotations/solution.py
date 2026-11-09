import numpy as np


def annotations(X, Wf, Uf, Wb, Ub):
    """Return the bidirectional annotations h_j = [forward_j; backward_j].

    X: (Tx, m); Wf, Wb: (n, m); Uf, Ub: (n, n)
    Returns an array of shape (Tx, 2n).
    """
    Tx, n = X.shape[0], Wf.shape[0]
    fwd = np.zeros((Tx, n))
    bwd = np.zeros((Tx, n))
    prev = np.zeros(n)
    for t in range(Tx):
        prev = np.tanh(Wf @ X[t] + Uf @ prev)
        fwd[t] = prev
    prev = np.zeros(n)
    for t in reversed(range(Tx)):
        prev = np.tanh(Wb @ X[t] + Ub @ prev)
        bwd[t] = prev
    return np.concatenate([fwd, bwd], axis=1)
