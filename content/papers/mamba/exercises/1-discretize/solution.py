import numpy as np


def discretize(delta, A, B):
    """Zero-order-hold discretization for a diagonal A (Equation 4, Section 2).

    delta: (L, D) step sizes, A: (D, N) diagonal entries (negative), B: (L, N).
    Returns (A_bar, B_bar), each of shape (L, D, N).
    """
    delta_A = delta[:, :, None] * A[None, :, :]
    A_bar = np.exp(delta_A)
    B_bar = np.expm1(delta_A) / A[None, :, :] * B[:, None, :]
    return A_bar, B_bar
