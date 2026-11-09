#!/usr/bin/env python3
"""One reflow round on a 2-D toy with the non-parametric estimator (Equation 5).

Noise pi_0 = N(0, I) in 2-D and data pi_1 = two blobs at (+-3, 0), coupled
independently, so the straight lines between pairs cross. The velocity field is
the Nadaraya-Watson estimator of Section 2.2, Equation 5, over all n pairs
(the paper restricts it to the m nearest neighbours, Section 5.1):

    v(z, t) = sum_i w_i (x1_i - z) / (1 - t),
    w_i  proportional to  exp(-|t x1_i + (1 - t) x0_i - z|^2 / (2 h^2)).

Reflow (Algorithm 4) simulates that flow from each x0_i and trains again on the
new pairs (x0_i, z1_i). This script prints the transport cost, the straightness
S(Z) of Equation 3, and the Euler error at few steps for flow 1 and flow 2.

Run from the repository root:

    python3 content/papers/rectified-flow/scripts/toy_reflow_2d.py
"""

import numpy as np

N_PAIRS = 400
H = 0.25
N_REF = 100  # Euler steps that stand in for the exact flow


def make_pairs(seed=0, n=N_PAIRS):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n, 2))
    centers = np.array([[3.0, 0.0], [-3.0, 0.0]])
    x1 = centers[rng.integers(0, 2, size=n)] + 0.3 * rng.normal(size=(n, 2))
    return x0, x1


def velocity(z, t, x0, x1, h=H):
    """Equation 5 with all pairs: kernel weights centered at x_t^(i)."""
    xt = t * x1 + (1.0 - t) * x0
    d2 = ((z[:, None, :] - xt[None]) ** 2).sum(-1)
    logw = -0.5 * d2 / h**2
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return (w @ x1 - z) / (1.0 - t)


def euler_path(x0, x1, n_steps, z0):
    """Euler trajectory of the flow induced by the pairs (x0, x1); shape (n_steps + 1, n, d)."""
    h = 1.0 / n_steps
    path = [z0]
    z = z0
    for k in range(n_steps):
        z = z + h * velocity(z, k * h, x0, x1)
        path.append(z)
    return np.stack(path)


def straightness(path):
    """Equation 3 on a uniform time grid: sum_k h * mean |(Z1 - Z0) - (Z_{k+1} - Z_k) / h|^2."""
    n_steps = len(path) - 1
    h = 1.0 / n_steps
    chord = path[-1] - path[0]
    rate = (path[1:] - path[:-1]) / h
    return float(h * ((chord[None] - rate) ** 2).sum(-1).mean(0).sum())


def main():
    x0, x1 = make_pairs()
    cost0 = float(((x1 - x0) ** 2).sum(-1).mean())

    path1 = euler_path(x0, x1, N_REF, x0)
    z1 = path1[-1]
    cost1 = float(((z1 - x0) ** 2).sum(-1).mean())

    path2 = euler_path(x0, z1, N_REF, x0)
    z2 = path2[-1]
    cost2 = float(((z2 - x0) ** 2).sum(-1).mean())

    print(f"pairs n = {N_PAIRS}, bandwidth h = {H}, reference steps = {N_REF}")
    print(f"transport cost E|X1 - X0|^2 (independent pairs): {cost0:.4f}")
    print(f"transport cost E|Z1 - Z0|^2 (1-rectified)      : {cost1:.4f}")
    print(f"transport cost E|Z1 - Z0|^2 (2-rectified)      : {cost2:.4f}")
    print(f"straightness S(Z^1) = {straightness(path1):.4f}")
    print(f"straightness S(Z^2) = {straightness(path2):.4f}")

    print("\nEuler error E|Z1^(N) - Z1^(100)|^2 at few steps (the flow's own 100-step endpoint):")
    print("  N   1-rectified   2-rectified")
    for n in (1, 2, 4, 8):
        e1 = float(((euler_path(x0, x1, n, x0)[-1] - z1) ** 2).sum(-1).mean())
        e2 = float(((euler_path(x0, z1, n, x0)[-1] - z2) ** 2).sum(-1).mean())
        print(f"{n:3d}   {e1:11.4f}   {e2:11.4f}")

    one1 = euler_path(x0, x1, 1, x0)[-1]
    print(f"\nOne Euler step of flow 1 lands at mean ({one1[:,0].mean():+.3f}, {one1[:,1].mean():+.3f}) "
          f"with x-spread {one1[:,0].std():.3f}; the data blobs sit at x = +-3.")


if __name__ == "__main__":
    main()
