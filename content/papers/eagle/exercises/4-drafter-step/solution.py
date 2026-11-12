import numpy as np


def drafter_input(F, tokens, E):
    """Return shape (n - 1, 2 d): row i is [F[i], E[tokens[i + 1]]]."""
    F = np.asarray(F)
    tokens = np.asarray(tokens)
    return np.concatenate([F[:-1], np.asarray(E)[tokens[1:]]], axis=1)


def drafter_step(F, tokens, E, W, b):
    """Return drafter_input(F, tokens, E) @ W + b, shape (n - 1, d)."""
    return drafter_input(F, tokens, E) @ W + b


def smooth_l1(pred, target, beta=1.0):
    """Mean Smooth L1 loss over all elements."""
    x = np.abs(np.asarray(pred, dtype=float) - np.asarray(target, dtype=float))
    return float(np.where(x < beta, 0.5 * x**2 / beta, x - 0.5 * beta).mean())
