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
    step = 1.0 / n_steps
    z = np.array(x0, dtype=float)
    path = [z]
    for k in range(n_steps):
        z = z + step * kernel_velocity(z, k * step, x0, x1, h)
        path.append(z)
    return np.stack(path)


def reflow(x0, x1, h, n_steps):
    """One round of Algorithm 4: simulate the flow and return the new pairs (x0, z1)."""
    return np.array(x0, dtype=float), euler_path(x0, x1, h, n_steps)[-1]
