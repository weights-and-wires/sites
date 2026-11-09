import numpy as np


def ctc_forward(y, label, blank=0):
    """Return (alpha, p) for the labelling `label` and the output matrix y.

    y: array of shape (T, K); y[t, k] is the probability of symbol k at frame t.
    label: tuple of integers that are not the blank.
    alpha: array of shape (T, 2 * len(label) + 1), unscaled (Equations 5 to 7).
    p: p(label | x), from Equation 8.
    """
    T = y.shape[0]
    lp = [blank]
    for k in label:
        lp += [k, blank]
    S = len(lp)
    alpha = np.zeros((T, S))
    alpha[0, 0] = y[0, blank]
    if S > 1:
        alpha[0, 1] = y[0, lp[1]]
    for t in range(1, T):
        for s in range(S):
            total = alpha[t - 1, s]
            if s >= 1:
                total += alpha[t - 1, s - 1]
            if s >= 2 and lp[s] != blank and lp[s] != lp[s - 2]:
                total += alpha[t - 1, s - 2]
            alpha[t, s] = total * y[t, lp[s]]
    p = alpha[T - 1, S - 1] + (alpha[T - 1, S - 2] if S > 1 else 0.0)
    return alpha, float(p)
