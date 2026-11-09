import numpy as np


def newton_schulz(G, steps=5, coeffs=(3.4445, -4.7750, 2.0315), eps=1e-7):
    """Finished for you: approximate U V^T for G = U S V^T."""
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


def muon_step(W, G, B, lr, mu):
    """Return (W_new, B_new). Don't modify W, G, or B."""
    B_new = mu * B + G
    O = newton_schulz(B_new)
    return W - lr * O, B_new
