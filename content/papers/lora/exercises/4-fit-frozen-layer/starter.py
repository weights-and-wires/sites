import numpy as np


def lora_forward(X, W0, A, B, alpha):
    """Finished for you: X W0^T + (alpha / r) X A^T B^T."""
    r = A.shape[0]
    return X @ W0.T + (alpha / r) * (X @ A.T) @ B.T


def lora_grads(X, A, B, alpha, dH):
    """Finished for you: gradients of the loss with respect to A and B."""
    s = alpha / A.shape[0]
    Z = X @ A.T
    dB = s * dH.T @ Z
    dA = (s * dH @ B).T @ X
    return dA, dB


def trainable_params(d, k, r):
    """Number of weights in A (r x k) and B (d x r)."""
    raise NotImplementedError


def train_lora(X, Y, W0, r, alpha, steps, lr, seed=0):
    """Train A and B by gradient descent on (1 / 2n) ||H - Y||_F^2.

    Start from A ~ N(0, 0.1^2) and B = 0. Return (A, B, losses), where losses
    holds the loss before step 1 and after every step (steps + 1 values).
    """
    raise NotImplementedError
