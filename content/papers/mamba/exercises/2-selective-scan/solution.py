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
    A_bar, B_bar = discretize(delta_t[None, :], A, B_t[None, :])
    h_new = A_bar[0] * h + B_bar[0] * x_t[:, None]
    y_t = (h_new * C_t[None, :]).sum(axis=-1)
    return y_t, h_new


def selective_scan(x, delta, A, B, C):
    """Run selective_step over a whole sequence, starting from h = 0.

    x: (L, D), delta: (L, D), A: (D, N), B: (L, N), C: (L, N). Returns y: (L, D).
    """
    L, D = x.shape
    h = np.zeros(A.shape)
    ys = []
    for t in range(L):
        y_t, h = selective_step(h, x[t], delta[t], A, B[t], C[t])
        ys.append(y_t)
    return np.stack(ys)
