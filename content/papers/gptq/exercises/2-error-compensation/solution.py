import numpy as np


def compensate(w, Hinv, q, wq_hat):
    """Return w after setting w[q] to wq_hat and updating all weights as in Equation 2."""
    delta = -(w[q] - wq_hat) / Hinv[q, q] * Hinv[:, q]
    return w + delta
