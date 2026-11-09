import numpy as np


def switch_route(x, Wr):
    """Top-1 routing.

    x:  (T, d_model) token representations.
    Wr: (d_model, N) router weights.

    Returns (p, idx, gate):
      p    (T, N) softmax over experts, Equation 1
      idx  (T,)   index of the expert with the highest probability
      gate (T,)   p[t, idx[t]], the value that scales the expert output
    """
    h = x @ Wr
    shifted = h - np.max(h, axis=-1, keepdims=True)
    e = np.exp(shifted)
    p = e / np.sum(e, axis=-1, keepdims=True)
    idx = np.argmax(p, axis=-1)
    gate = p[np.arange(len(p)), idx]
    return p, idx, gate
