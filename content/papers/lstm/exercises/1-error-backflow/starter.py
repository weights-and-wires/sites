import numpy as np


def rnn_jacobians(W, U, xs, h0):
    """Return an array J of shape (T, n, n) with J[t-1] = d h_t / d h_0.

    The network is h_t = tanh(W h_{t-1} + U x_t), with xs of shape (T, d_in).
    """
    raise NotImplementedError


def cec_jacobians(n, T):
    """Return an array of shape (T, n, n) with d s_t / d s_0 for a bank of n CECs.

    Each unit has the identity as its activation function and a self-connection of 1.0.
    """
    raise NotImplementedError


def lag_below(norms, threshold):
    """Return the smallest lag q (counting from 1) with norms[q - 1] < threshold, or None."""
    raise NotImplementedError
