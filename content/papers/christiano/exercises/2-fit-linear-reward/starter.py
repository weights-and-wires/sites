import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def loss_grad(w, F1, F2, mu):
    """Return d(mean cross-entropy)/dw, shape (d,). F1 and F2 have shape (n, k, d)."""
    raise NotImplementedError


def fit_reward(F1, F2, mu, lr=0.5, steps=400):
    """Full-batch gradient descent from w = 0. Return w, shape (d,)."""
    raise NotImplementedError
