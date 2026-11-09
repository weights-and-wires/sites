import numpy as np


def discretize(delta, A, B):
    """Zero-order-hold discretization (Equation 4); given, see Exercise 1.

    delta: (L, D), A: (D, N), B: (L, N). Returns (A_bar, B_bar), each (L, D, N).
    """
    delta_A = delta[:, :, None] * A[None, :, :]
    return np.exp(delta_A), np.expm1(delta_A) / A[None, :, :] * B[:, None, :]


def selective_step(h, x_t, delta_t, A, B_t, C_t):
    """One step of Algorithm 2. Returns (y_t, h_new).

    h: (D, N), x_t: (D,), delta_t: (D,), A: (D, N), B_t: (N,), C_t: (N,).
    y_t has shape (D,).
    """
    raise NotImplementedError


def selective_scan(x, delta, A, B, C):
    """Run selective_step over a whole sequence, starting from h = 0.

    x: (L, D), delta: (L, D), A: (D, N), B: (L, N), C: (L, N). Returns y: (L, D).
    """
    raise NotImplementedError
