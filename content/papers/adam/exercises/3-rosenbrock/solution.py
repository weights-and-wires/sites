import warnings

import numpy as np

warnings.filterwarnings("ignore")


def rosenbrock(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def rosenbrock_grad(p):
    """Gradient of the Rosenbrock function at p = [x, y]; shape (2,)."""
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])


def sgd_momentum(grad_fn, x0, steps, lr, mu=0.9):
    """Heavy-ball SGD: v = mu * v + g, then x = x - lr * v. Returns the final x."""
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    for _ in range(steps):
        v = mu * v + grad_fn(x)
        x = x - lr * v
    return x


def adam(grad_fn, x0, steps, alpha, beta1=0.9, beta2=0.999, eps=1e-8):
    """Finished for you: Algorithm 1 from Section 2."""
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, steps + 1):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        x = x - alpha * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
    return x
