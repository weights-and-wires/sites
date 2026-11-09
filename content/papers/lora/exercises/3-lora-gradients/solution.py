import numpy as np


def lora_forward(X, W0, A, B, alpha):
    """Finished for you: X W0^T + (alpha / r) X A^T B^T."""
    r = A.shape[0]
    return X @ W0.T + (alpha / r) * (X @ A.T) @ B.T


def lora_grads(X, A, B, alpha, dH):
    """Return (dA, dB), the gradients of the loss with respect to A and B.

    X is (n, k), A is (r, k), B is (d, r), dH is (n, d).
    """
    s = alpha / A.shape[0]
    Z = X @ A.T
    dB = s * dH.T @ Z
    dZ = s * dH @ B
    dA = dZ.T @ X
    return dA, dB
