import numpy as np


def drafter_input(F, tokens, E):
    """Return shape (n - 1, 2 d): row i is [F[i], E[tokens[i + 1]]]."""
    raise NotImplementedError


def drafter_step(F, tokens, E, W, b):
    """Return drafter_input(F, tokens, E) @ W + b, shape (n - 1, d)."""
    raise NotImplementedError


def smooth_l1(pred, target, beta=1.0):
    """Mean Smooth L1 loss over all elements."""
    raise NotImplementedError
