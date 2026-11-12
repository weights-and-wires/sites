"""Numbers for "Overestimation bias" (Section 4, and the Section 4.2 remark on variance).

All true action values are 0. Each estimate is the truth plus N(0, sigma) noise.

Run: python3 scripts/overestimation.py
"""

import math

import numpy as np


def exact_expected_max(n, sigma=1.0):
    """E[max of n iid N(0, sigma)] by numerical integration of x * n * phi(x) * Phi(x)^(n-1)."""
    xs = np.linspace(-10, 10, 200001)
    phi = np.exp(-0.5 * xs**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(xs / math.sqrt(2)))
    return float(sigma * np.sum(xs * n * phi * cdf ** (n - 1)) * (xs[1] - xs[0]))


def bias(n_actions, sigma, trials, seed):
    rng = np.random.default_rng(seed)
    qa = rng.normal(0.0, sigma, (trials, n_actions))
    qb = rng.normal(0.0, sigma, (trials, n_actions))
    star = qa.argmax(axis=1)
    rows = np.arange(trials)
    return {
        "max": qa.max(axis=1).mean(),
        "double": qb[rows, star].mean(),
        "clipped": np.minimum(qa[rows, star], qb[rows, star]).mean(),
    }


print("Exact E[max] of standard-normal estimates (true max is 0):")
for n in (2, 5, 10):
    print(f"  {n:2d} actions: {exact_expected_max(n):.4f}")
print(f"  check, n = 2: 1/sqrt(pi) = {1 / math.sqrt(math.pi):.4f}")

print("Monte Carlo, 200000 trials, sigma = 1 (seed 0):")
for n in (2, 10):
    b = bias(n, 1.0, 200000, 0)
    print(f"  {n:2d} actions: max {b['max']:+.4f}  double {b['double']:+.4f}  clipped {b['clipped']:+.4f}")

print("Scaling with the noise level, 10 actions (seed 0):")
for sigma in (0.5, 1.0, 2.0):
    b = bias(10, sigma, 200000, 0)
    print(f"  sigma {sigma}: max {b['max']:+.4f}  clipped {b['clipped']:+.4f}")

print("Expected min of two independent N(0, s) estimates is -s / sqrt(pi):")
for s in (0.5, 1.0, 2.0):
    print(f"  s = {s}: {-s / math.sqrt(math.pi):+.4f}")
