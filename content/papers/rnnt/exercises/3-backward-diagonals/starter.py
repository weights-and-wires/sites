import numpy as np


def forward(emit, null):
    """Given: alpha (Equation 16) and Pr(y | x) (Equation 17)."""
    T, U1 = null.shape
    alpha = np.zeros((T, U1))
    alpha[0, 0] = 1.0
    for t in range(T):
        for u in range(U1):
            if t == 0 and u == 0:
                continue
            from_left = alpha[t - 1, u] * null[t - 1, u] if t > 0 else 0.0
            from_below = alpha[t, u - 1] * emit[t, u - 1] if u > 0 else 0.0
            alpha[t, u] = from_left + from_below
    return alpha, alpha[T - 1, U1 - 1] * null[T - 1, U1 - 1]


def backward(emit, null):
    """Return beta of shape (T, U+1) (Equation 18), with beta(T, U) = null(T, U)."""
    raise NotImplementedError


def diagonal_sums(alpha, beta):
    """Return an array of length T + U whose entry n - 1 is the sum of
    alpha * beta over the nodes with t + u = n (1-based t, Equation 19)."""
    raise NotImplementedError
