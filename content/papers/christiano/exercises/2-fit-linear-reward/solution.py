import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def loss_grad(w, F1, F2, mu):
    """Return d(mean cross-entropy)/dw, shape (d,). F1 and F2 have shape (n, k, d)."""
    S1 = F1.sum(axis=1)
    S2 = F2.sum(axis=1)
    diff = (S1 - S2) @ w
    return ((sigmoid(diff) - mu)[:, None] * (S1 - S2)).mean(axis=0)


def fit_reward(F1, F2, mu, lr=0.5, steps=400):
    """Full-batch gradient descent from w = 0. Return w, shape (d,)."""
    w = np.zeros(F1.shape[-1])
    for _ in range(steps):
        w = w - lr * loss_grad(w, F1, F2, mu)
    return w
