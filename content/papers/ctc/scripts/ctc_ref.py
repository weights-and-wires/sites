"""Reference CTC code shared by the example scripts.

Index 0 is the blank. Labels are integers 1..K-1. The functions follow the
paper's notation: a path pi, the map B, the forward variable alpha (Equations
5 to 8), and the backward variable beta (Equations 9 to 11). The code indexes
time and label positions from 0, where the paper starts at 1.
"""

import itertools

import numpy as np

BLANK = 0


def collapse(path):
    """The map B: merge repeated symbols, then drop blanks."""
    out = []
    prev = None
    for s in path:
        if s != prev and s != BLANK:
            out.append(s)
        prev = s
    return tuple(out)


def brute_force(y, label):
    """Return p(l|x) by enumerating every path (Equations 2 and 3), and the matching paths."""
    T, K = y.shape
    total = 0.0
    hits = []
    for path in itertools.product(range(K), repeat=T):
        if collapse(path) == tuple(label):
            p = float(np.prod([y[t, k] for t, k in enumerate(path)]))
            total += p
            hits.append((path, p))
    return total, hits


def extend(label):
    """The modified label sequence l' with blanks at both ends and between labels."""
    out = [BLANK]
    for k in label:
        out += [k, BLANK]
    return out


def forward(y, label):
    """Return alpha with shape (T, 2|l|+1), unscaled (Equations 5 to 7)."""
    T = y.shape[0]
    lp = extend(label)
    S = len(lp)
    a = np.zeros((T, S))
    a[0, 0] = y[0, BLANK]
    if S > 1:
        a[0, 1] = y[0, lp[1]]
    for t in range(1, T):
        for s in range(S):
            tot = a[t - 1, s] + (a[t - 1, s - 1] if s >= 1 else 0.0)
            if s >= 2 and lp[s] != BLANK and lp[s] != lp[s - 2]:
                tot += a[t - 1, s - 2]
            a[t, s] = tot * y[t, lp[s]]
    return a


def backward(y, label):
    """Return beta with shape (T, 2|l|+1), unscaled (Equations 9 to 11)."""
    T = y.shape[0]
    lp = extend(label)
    S = len(lp)
    b = np.zeros((T, S))
    b[T - 1, S - 1] = y[T - 1, BLANK]
    if S > 1:
        b[T - 1, S - 2] = y[T - 1, lp[S - 2]]
    for t in range(T - 2, -1, -1):
        for s in range(S):
            tot = b[t + 1, s] + (b[t + 1, s + 1] if s + 1 < S else 0.0)
            if s + 2 < S and lp[s] != BLANK and lp[s] != lp[s + 2]:
                tot += b[t + 1, s + 2]
            b[t, s] = tot * y[t, lp[s]]
    return b


def prob(y, label):
    """p(l|x) from Equation 8."""
    a = forward(y, label)
    S = a.shape[1]
    return a[-1, S - 1] + (a[-1, S - 2] if S > 1 else 0.0)


def grad_u(y, label):
    """Gradient of -ln p(l|x) with respect to the unnormalized outputs (Equation 16)."""
    a, b = forward(y, label), backward(y, label)
    lp = extend(label)
    p = prob(y, label)
    g = np.zeros_like(y, dtype=float)
    for t in range(y.shape[0]):
        acc = np.zeros(y.shape[1])
        for s, k in enumerate(lp):
            acc[k] += a[t, s] * b[t, s]
        g[t] = y[t] - acc / (y[t] * p)
    return g


def best_path(y):
    """Best path decoding (Section 3.2, Equation 4)."""
    return collapse(y.argmax(axis=1))
