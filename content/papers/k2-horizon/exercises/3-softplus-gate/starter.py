import numpy as np


def softplus(x, beta):
    """Equation 9: (1 / beta) * log(1 + exp(beta * x)), without overflow."""
    raise NotImplementedError


def gated_output(Y, X, W_g, b_g, beta=np.log(2)):
    """Equation 8 with the softplus gate: Y * softplus(X W_g + b_g)."""
    raise NotImplementedError
