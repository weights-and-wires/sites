import numpy as np


def isoflop_minimum(sizes, losses):
    """Fit loss = c2*x**2 + c1*x + c0 with x = ln(size) and return (n_best, loss_min).

    n_best is the model size at the vertex; loss_min is the fitted loss there.
    """
    raise NotImplementedError
