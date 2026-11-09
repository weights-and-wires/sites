#!/usr/bin/env python3
"""The exact rectified flow on a 1-D example: marginals, cost, and one-step collapse.

Noise X0 ~ N(0, 1) and data X1 = -2 or +2 with probability 1/2 each, independent.
Everything below uses the paper's notation (Section 2.1): X_t = t X1 + (1 - t) X0,
v^X(x, t) = E[X1 - X0 | X_t = x] (Equation 2).

Given X1 = a, X0 = (x - t a) / (1 - t), so X1 - X0 = (a - x) / (1 - t) and

    v^X(x, t) = (E[a | X_t = x] - x) / (1 - t),
    E[a | X_t = x] = 2 tanh(2 t x / (1 - t)^2),

because the posterior odds of a = +2 against a = -2 are exp(4 t x / (1 - t)^2).
This is Equation 4 of Section 2.2 with a Gaussian conditional density rho.

The script integrates dZ_t = v^X(Z_t, t) dt with RK4 on a graded time grid, and
prints (Theorem 3.3) the marginals of Z_t against X_t, (Theorem 3.5) the
transport costs, and the one-Euler-step endpoint.

Run from the repository root:

    python3 content/papers/rectified-flow/scripts/rewire_1d.py
"""

import math

import numpy as np

A = 2.0  # the two data points are -A and +A


def v_x(x, t):
    """Equation 2 in closed form for this example."""
    return (A * np.tanh(A * t * x / (1.0 - t) ** 2) - x) / (1.0 - t)


def gaussian_grid(n=20000, lim=8.0):
    """Quadrature nodes and weights for X0 ~ N(0, 1)."""
    z0 = np.linspace(-lim, lim, n)
    w = np.exp(-0.5 * z0**2)
    return z0, w / w.sum()


def graded_times(n_steps, power=3):
    """Time grid that is dense near t = 1, where 1 / (1 - t) makes the field stiff."""
    u = np.linspace(0.0, 1.0, n_steps + 1)
    return 1.0 - (1.0 - u) ** power


def rk4_path(v, z0, times):
    """Return the trajectory at every grid time, stopping one node before t = 1."""
    z = z0.copy()
    out = [z]
    for k in range(len(times) - 2):  # the last node is t = 1 exactly: skip it
        t, h = times[k], times[k + 1] - times[k]
        k1 = v(z, t)
        k2 = v(z + 0.5 * h * k1, t + 0.5 * h)
        k3 = v(z + 0.5 * h * k2, t + 0.5 * h)
        k4 = v(z + h * k3, t + h)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(z)
    return np.stack(out), times[: len(out)]


def rk4_to(v, z0, t_end, n_steps):
    """RK4 with uniform steps from t = 0 to t_end < 1."""
    z, h = z0.copy(), t_end / n_steps
    for k in range(n_steps):
        t = k * h
        k1 = v(z, t)
        k2 = v(z + 0.5 * h * k1, t + 0.5 * h)
        k3 = v(z + 0.5 * h * k2, t + 0.5 * h)
        k4 = v(z + h * k3, t + h)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def x_t_moments(t):
    """Variance and E|X_t| of X_t = t X1 + (1 - t) X0 in closed form."""
    s = 1.0 - t
    var = (t * A) ** 2 + s**2
    m = t * A
    e_abs = s * math.sqrt(2 / math.pi) * math.exp(-0.5 * (m / s) ** 2) + m * math.erf(m / (s * math.sqrt(2)))
    return var, e_abs


def main():
    z0, w = gaussian_grid()
    times = graded_times(2000)
    path, tt = rk4_path(v_x, z0, times)
    z1 = A * np.sign(z0)  # the flow ends at the nearer data point; checked below
    end = path[-1]
    print(f"largest |Z_(1-eps) - 2 sign(Z_0)| over the grid: {np.abs(end - z1)[np.abs(z0) > 1e-3].max():.2e}")

    print("\nTheorem 3.3: Law(Z_t) = Law(X_t). Variance and E|.| at four times")
    print("   t   Var X_t   Var Z_t   E|X_t|   E|Z_t|")
    for t_want in (0.25, 0.5, 0.75, 0.9):
        zt = rk4_to(v_x, z0, t_want, 800)
        var_x, abs_x = x_t_moments(t_want)
        var_z = float((w * zt**2).sum())
        abs_z = float((w * np.abs(zt)).sum())
        print(f"{t_want:5.3f}   {var_x:7.4f}   {var_z:7.4f}   {abs_x:6.4f}   {abs_z:6.4f}")

    print("\nTheorem 3.5: transport costs, independent pairs against the rectified pairs")
    cost_sq_x = A**2 + 1.0  # E(X1 - X0)^2 = E X1^2 + E X0^2 for independent zero-mean variables
    cost_sq_z = float((w * (z1 - z0) ** 2).sum())
    cost_abs_x = float(
        0.5 * (math.sqrt(2 / math.pi) * math.exp(-0.5 * A**2) + A * math.erf(A / math.sqrt(2))) * 2
    )
    cost_abs_z = float((w * np.abs(z1 - z0)).sum())
    print(f"c(x) = x^2 : E c(X1 - X0) = {cost_sq_x:.4f}   E c(Z1 - Z0) = {cost_sq_z:.4f}"
          f"   closed form 5 - 4 sqrt(2/pi) = {5 - 4 * math.sqrt(2 / math.pi):.4f}")
    print(f"c(x) = |x| : E c(X1 - X0) = {cost_abs_x:.4f}   E c(Z1 - Z0) = {cost_abs_z:.4f}")

    print("\nOne Euler step, Z_1 = Z_0 + v(Z_0, 0):")
    one = z0 + v_x(z0, 0.0)
    print(f"v(x, 0) = -x, so every sample lands at {float(np.abs(one).max()):.1f}: the mean of pi_1, not a data point.")

    print("\nThe field at x = 0.5, t = 0.5 (Equation 2):")
    print(f"v^X(0.5, 0.5) = {float(v_x(np.array(0.5), 0.5)):.4f}")
    print(f"E[a | X_t = 0.5] = {float(A * np.tanh(A * 0.5 * 0.5 / 0.25)):.4f}")


if __name__ == "__main__":
    main()
