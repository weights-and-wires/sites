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
    return r * (k + d)


def train_lora(X, Y, W0, r, alpha, steps, lr, seed=0):
    """Train A and B by gradient descent on (1 / 2n) ||H - Y||_F^2.

    Start from A ~ N(0, 0.1^2) and B = 0. Return (A, B, losses), where losses
    holds the loss before step 1 and after every step (steps + 1 values).
    """
    d, k = W0.shape
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 0.1, size=(r, k))
    B = np.zeros((d, r))
    losses = []
    for _ in range(steps):
        H = lora_forward(X, W0, A, B, alpha)
        R = H - Y
        losses.append(float(np.sum(R**2) / (2 * n)))
        dA, dB = lora_grads(X, A, B, alpha, R / n)
        A = A - lr * dA
        B = B - lr * dB
    H = lora_forward(X, W0, A, B, alpha)
    losses.append(float(np.sum((H - Y) ** 2) / (2 * n)))
    return A, B, losses
