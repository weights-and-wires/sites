import numpy as np


def kernel_velocity(z, t, x0, x1, h):
    """Nadaraya-Watson estimate of the rectified velocity field (Equation 5).

    z has shape (m, d): the points where the field is evaluated.
    x0 and x1 have shape (n, d): paired draws of (X0, X1).
    t is a float in [0, 1) and h > 0 is the kernel bandwidth.
    Returns an array of shape (m, d).
    """
    raise NotImplementedError
