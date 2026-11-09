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
    raise NotImplementedError
