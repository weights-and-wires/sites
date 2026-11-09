"""Measure how far a collapsed generator is from a two-mode data distribution.

The data are an equal mixture of N(-2, 0.5^2) and N(2, 0.5^2). We compare
histogram-based JS divergences (natural log) for three generators:
a perfect one, one that collapses onto a single mode, and the best single Gaussian.

Run: python3 scripts/mixture_collapse.py
"""

import numpy as np

rng = np.random.default_rng(0)
edges = np.linspace(-5, 5, 41)


def hist(x):
    h, _ = np.histogram(x, bins=edges)
    return h / h.sum()


def jsd(p, q):
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mixture(n):
    comp = rng.integers(0, 2, size=n)
    return np.where(comp == 0, -2.0, 2.0) + 0.5 * rng.normal(size=n)


n = 100_000
data = mixture(n)
perfect = mixture(n)
collapsed = 2.0 + 0.5 * rng.normal(size=n)
best_single = data.mean() + data.std() * rng.normal(size=n)

print("JSD(data, perfect generator)       =", round(float(jsd(hist(data), hist(perfect))), 4))
print("JSD(data, collapsed onto +2)       =", round(float(jsd(hist(data), hist(collapsed))), 4))
print("JSD(data, best single Gaussian)    =", round(float(jsd(hist(data), hist(best_single))), 4))
print("log 2 (the maximum)                =", round(float(np.log(2)), 4))
print("single Gaussian mean, std          =", round(float(data.mean()), 3), round(float(data.std()), 3))
for name, s in [("data", data), ("collapsed", collapsed)]:
    near = (np.abs(s - 2.0) < 1.0).mean(), (np.abs(s + 2.0) < 1.0).mean()
    print(f"share near +2 and -2 ({name:9s}) =", round(float(near[0]), 3), round(float(near[1]), 3))
