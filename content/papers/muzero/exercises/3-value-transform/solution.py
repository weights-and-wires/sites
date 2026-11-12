import numpy as np

EPS = 0.001


def h(x):
    """Return sign(x) * (sqrt(|x| + 1) - 1) + EPS * x."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.0) - 1.0) + EPS * x


def h_inv(y):
    """Return the exact inverse of h."""
    y = np.asarray(y, dtype=float)
    w = (np.sqrt(1.0 + 4.0 * EPS * (np.abs(y) + 1.0 + EPS)) - 1.0) / (2.0 * EPS)
    return np.sign(y) * (w ** 2 - 1.0)


def phi(x, lo=-300, hi=300):
    """Return the weights over the integer support lo..hi (length hi - lo + 1)."""
    x = float(np.clip(x, lo, hi))
    out = np.zeros(hi - lo + 1)
    low = int(np.floor(x))
    frac = x - low
    out[low - lo] = 1.0 - frac
    if frac > 0.0:
        out[low + 1 - lo] = frac
    return out


def to_target(x, lo=-300, hi=300):
    """Return phi(h(x)): the categorical training target for a scalar x."""
    return phi(h(x), lo, hi)


def from_logits(logits, lo=-300):
    """Softmax the logits, take the expected support value, then invert h."""
    logits = np.asarray(logits, dtype=float)
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    support = lo + np.arange(len(logits))
    return float(h_inv(p @ support))
