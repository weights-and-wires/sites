import numpy as np


def row_convolution(h, W):
    """Row convolution (Equation 11). `h` has shape (T, d) and `W` has shape (d, tau + 1).

    Return r with shape (T, d), where r[t, i] = sum_j W[i, j] * h[t + j, i] (0-based j),
    and h[t + j, i] is 0 when t + j >= T.
    """
    h = np.asarray(h, dtype=float)
    W = np.asarray(W, dtype=float)
    T, d = h.shape
    tau = W.shape[1] - 1
    padded = np.vstack([h, np.zeros((tau, d))])
    r = np.zeros((T, d))
    for j in range(tau + 1):
        r += W[:, j] * padded[j : j + T]
    return r
