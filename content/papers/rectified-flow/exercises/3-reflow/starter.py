import numpy as np


def kernel_velocity(z, t, x0, x1, h):
    """Equation 5 over all pairs (given; this is Exercise 1)."""
    z = np.asarray(z, dtype=float)
    xt = t * x1 + (1.0 - t) * x0
    d2 = ((z[:, None, :] - xt[None, :, :]) ** 2).sum(axis=-1)
    logw = -0.5 * d2 / h**2
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return (w @ x1 - z) / (1.0 - t)


def euler_path(x0, x1, h, n_steps):
    """Euler trajectories of the rectified flow induced by the pairs (x0, x1).

    Start each trajectory at its own x0. Step k uses the field at time
    k / n_steps. Returns an array of shape (n_steps + 1, n, d).
    """
    raise NotImplementedError


def reflow(x0, x1, h, n_steps):
    """One round of Algorithm 4: simulate the flow and return the new pairs (x0, z1)."""
    raise NotImplementedError
