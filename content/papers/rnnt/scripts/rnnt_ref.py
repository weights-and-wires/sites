"""Reference implementation of the RNN transducer lattice (Graves 2012, Section 2).

Conventions. Index 0 is the null symbol and the labels are 1..K. The code is
0-based, so `alpha[t, u]` is the paper's alpha(t + 1, u). A target `y` is a
tuple of labels of length U. Arrays:

    f     (T, K+1)   transcription vectors f_t (one row per frame)
    g     (U+1, K+1) prediction vectors g_u (u = 0..U)
    emit  (T, U)     y(t, u)  = Pr(y_{u+1} | t, u)
    null  (T, U+1)   null(t, u) = Pr(null | t, u)

Not a script to run on its own; the other scripts import it.
"""

import itertools
import math

import numpy as np


def lattice_probs(f, g):
    """Pr(k | t, u) for every node, Equations 12 and 13. Shape (T, U+1, K+1)."""
    logits = f[:, None, :] + g[None, :, :]
    logits = logits - logits.max(axis=-1, keepdims=True)
    h = np.exp(logits)
    return h / h.sum(axis=-1, keepdims=True)


def emit_null(probs, y):
    """Split Pr(k | t, u) into y(t, u) (Equation 14) and null(t, u) (Equation 15)."""
    U = len(y)
    emit = np.stack([probs[:, u, y[u]] for u in range(U)], axis=1) if U else np.zeros((probs.shape[0], 0))
    return emit, probs[:, :, 0]


def forward(emit, null):
    """alpha(t, u) from Equation 16 and Pr(y | x) from Equation 17."""
    T, U1 = null.shape
    alpha = np.zeros((T, U1))
    alpha[0, 0] = 1.0
    for t in range(T):
        for u in range(U1):
            if t == 0 and u == 0:
                continue
            a = alpha[t - 1, u] * null[t - 1, u] if t > 0 else 0.0
            b = alpha[t, u - 1] * emit[t, u - 1] if u > 0 else 0.0
            alpha[t, u] = a + b
    return alpha, alpha[T - 1, U1 - 1] * null[T - 1, U1 - 1]


def backward(emit, null):
    """beta(t, u) from Equation 18, with beta(T, U) = null(T, U)."""
    T, U1 = null.shape
    beta = np.zeros((T, U1))
    for t in range(T - 1, -1, -1):
        for u in range(U1 - 1, -1, -1):
            if t == T - 1 and u == U1 - 1:
                beta[t, u] = null[t, u]
                continue
            a = beta[t + 1, u] * null[t, u] if t < T - 1 else 0.0
            b = beta[t, u + 1] * emit[t, u] if u < U1 - 1 else 0.0
            beta[t, u] = a + b
    return beta


def brute_force(emit, null):
    """Sum over every alignment. Returns (total, list of (alignment string, probability)).

    An alignment is a walk on the lattice from (1, 0) to (T, U) that ends with
    the null at (T, U). In the string, `e` is a label emission and `-` is a null.
    """
    T, U1 = null.shape
    U = U1 - 1
    out = []
    for emits_at in itertools.combinations(range(T + U - 1), U):
        t = u = 0
        p = 1.0
        s = ""
        for step in range(T + U - 1):
            if step in emits_at:
                p *= emit[t, u]
                u += 1
                s += "e"
            else:
                p *= null[t, u]
                t += 1
                s += "-"
        p *= null[T - 1, U]
        s += "-"
        out.append((s, p))
    return sum(p for _, p in out), out


def count_alignments(T, U):
    """Number of alignments: lattice paths from (1, 0) to (T, U)."""
    return math.comb(T + U - 1, U)


def gradients(f, g, y):
    """Gradient of L = -ln Pr(y | x) with respect to f and g (Equations 20 to 22)."""
    probs = lattice_probs(f, g)
    emit, null = emit_null(probs, y)
    alpha, total = forward(emit, null)
    beta = backward(emit, null)
    T, U1, K1 = probs.shape
    dP = np.zeros_like(probs)
    for t in range(T):
        for u in range(U1):
            nxt = beta[t + 1, u] if t < T - 1 else (1.0 if u == U1 - 1 else 0.0)
            dP[t, u, 0] = -alpha[t, u] * nxt / total
            if u < U1 - 1:
                dP[t, u, y[u]] = -alpha[t, u] * beta[t, u + 1] / total
    s = (dP * probs).sum(axis=-1, keepdims=True)
    d_logit = probs * (dP - s)
    return d_logit.sum(axis=1), d_logit.sum(axis=0), total


def neg_log_prob(f, g, y):
    emit, null = emit_null(lattice_probs(f, g), y)
    return -math.log(forward(emit, null)[1])
