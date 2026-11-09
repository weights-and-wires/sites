import numpy as np


def gelu(x):
    """The tanh approximation of GeLU."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def dgelu(x):
    """Derivative of the tanh approximation of GeLU."""
    c = np.sqrt(2 / np.pi)
    t = np.tanh(c * (x + 0.044715 * x**3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * c * (1 + 3 * 0.044715 * x**2)


def all_reduce(parts):
    """Finished for you: sum a list of per-GPU arrays; every GPU receives the sum."""
    return sum(parts)


def tp_mlp_backward(X, A, B, dZ, n):
    """Backward pass of Z = gelu(X A) B with A split by columns and B by rows.

    Returns (dX, dA_blocks, dB_blocks).
    """
    raise NotImplementedError
