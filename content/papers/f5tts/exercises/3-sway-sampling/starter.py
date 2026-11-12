import numpy as np


def sway(u, s):
    """Equation 7: u + s * (cos(pi / 2 * u) - 1 + u)."""
    raise NotImplementedError


def sway_steps(nfe, s):
    """Return the nfe + 1 flow-step boundaries as a NumPy array.

    Apply sway to np.linspace(0, 1, nfe + 1). Raise ValueError unless
    -1 <= s <= 2 / (pi - 2).
    """
    raise NotImplementedError
