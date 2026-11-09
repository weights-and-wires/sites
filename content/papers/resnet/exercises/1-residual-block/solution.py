import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def residual_block(x, W1, W2, Ws=None):
    """Return y = relu(F(x) + shortcut(x)) for the two-layer block in Figure 2.

    F(x) = W2 relu(W1 x), with no biases (Section 3.2). The shortcut is the
    identity (Equation 1) when Ws is None and Ws @ x otherwise (Equation 2).
    The second nonlinearity comes after the addition.
    """
    F = W2 @ relu(W1 @ x)
    shortcut = x if Ws is None else Ws @ x
    return relu(F + shortcut)
