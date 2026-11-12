import numpy as np


def sway(u, s):
    """Equation 7: u + s * (cos(pi / 2 * u) - 1 + u)."""
    u = np.asarray(u, dtype=float)
    return u + s * (np.cos(np.pi / 2 * u) - 1 + u)


def sway_steps(nfe, s):
    """Return the nfe + 1 flow-step boundaries as a NumPy array.

    Apply sway to np.linspace(0, 1, nfe + 1). Raise ValueError unless
    -1 <= s <= 2 / (pi - 2).
    """
    s_max = 2 / (np.pi - 2)
    if s < -1 - 1e-12 or s > s_max + 1e-12:
        raise ValueError(f"s = {s} is outside [-1, {s_max:.4f}]")
    return sway(np.linspace(0, 1, nfe + 1), s)
