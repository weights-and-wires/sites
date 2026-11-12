import itertools

import numpy as np


def forward(emit, null):
    """Return (alpha, prob).

    emit: (T, U), emit[t, u] = y(t+1, u).
    null: (T, U+1), null[t, u] = null(t+1, u).
    alpha: (T, U+1) forward variables (Equation 16).
    prob: Pr(y | x) (Equation 17).
    """
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


def brute_force(emit, null):
    """Return (prob, n): the sum over every alignment and how many there are."""
    T, U1 = null.shape
    U = U1 - 1
    total = 0.0
    n = 0
    for label_steps in itertools.combinations(range(T + U - 1), U):
        t = u = 0
        p = 1.0
        for step in range(T + U - 1):
            if step in label_steps:
                p *= emit[t, u]
                u += 1
            else:
                p *= null[t, u]
                t += 1
        p *= null[T - 1, U]
        total += p
        n += 1
    return total, n
