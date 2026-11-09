import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def block_forward(x, W1, W2):
    """Finished for you. y = relu(W2 relu(W1 x) + x), with an identity shortcut."""
    return relu(W2 @ relu(W1 @ x) + x)


def block_backward(x, W1, W2, g_out):
    """Return (g_x, gW1, gW2), the gradients of a loss L with respect to x, W1, and W2.

    g_out is dL/dy. Recompute whatever you need from the forward pass.
    """
    raise NotImplementedError
