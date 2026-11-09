import numpy as np


def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Approximate U V^T for G = U S V^T by `steps` quintic Newton-Schulz steps.

    G is a 2D float array. Return an array of the same shape. Don't modify G.
    """
    assert G.ndim == 2
    a, b, c = coeffs
    X = G / (np.linalg.norm(G) + eps)
    tall = G.shape[0] > G.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if tall:
        X = X.T
    return X
