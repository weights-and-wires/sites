import numpy as np


def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Approximate U V^T for G = U S V^T by `steps` quintic Newton-Schulz steps.

    G is a 2D float array. Return an array of the same shape. Don't modify G.
    """
    raise NotImplementedError
