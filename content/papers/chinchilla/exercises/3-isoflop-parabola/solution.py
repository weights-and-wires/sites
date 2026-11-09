import numpy as np


def isoflop_minimum(sizes, losses):
    """Fit loss = c2*x**2 + c1*x + c0 with x = ln(size) and return (n_best, loss_min).

    n_best is the model size at the vertex; loss_min is the fitted loss there.
    """
    x = np.log(np.asarray(sizes, dtype=float))
    c2, c1, c0 = np.polyfit(x, np.asarray(losses, dtype=float), 2)
    x_star = -c1 / (2 * c2)
    return float(np.exp(x_star)), float(np.polyval([c2, c1, c0], x_star))
