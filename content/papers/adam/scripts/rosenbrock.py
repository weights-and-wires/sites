"""Adam and SGD with momentum on the 2-D Rosenbrock function.

f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, minimum 0 at (1, 1).

Run: python3 scripts/rosenbrock.py
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")


def f(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def grad(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])


def adam(g_fn, x0, steps, alpha, beta1=0.9, beta2=0.999, eps=1e-8):
    x = np.array(x0, float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, steps + 1):
        g = g_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        x = x - alpha * (m / (1 - beta1**t)) / (np.sqrt(v / (1 - beta2**t)) + eps)
    return x


def momentum(g_fn, x0, steps, lr, mu=0.9):
    x = np.array(x0, float)
    v = np.zeros_like(x)
    for _ in range(steps):
        v = mu * v + g_fn(x)
        x = x - lr * v
    return x


x0 = [-1.5, 1.5]
print(f"start {x0}, f = {f(np.array(x0)):.4f}, 2000 steps each")
for alpha in (0.01, 0.05, 0.1):
    x = adam(grad, x0, 2000, alpha)
    print(f"adam     alpha={alpha:<6} f = {f(x):.3e}")
for lr in (1e-4, 3e-4, 1e-3, 3e-3):
    x = momentum(grad, x0, 2000, lr)
    print(f"momentum lr={lr:<8} f = {f(x):.3e}")

print()
print("Multiply the gradient by 1000 (the same problem in different units)")
scaled = lambda p: 1000 * grad(p)
x = adam(scaled, x0, 2000, 0.1)
print(f"adam     alpha=0.1    f = {f(x):.3e}")
x = momentum(scaled, x0, 2000, 1e-3)
print(f"momentum lr=0.001     f = {f(x):.3e}")
x = momentum(scaled, x0, 2000, 1e-6)
print(f"momentum lr=1e-06     f = {f(x):.3e}")
