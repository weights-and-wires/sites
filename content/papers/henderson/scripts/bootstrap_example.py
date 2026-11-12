"""Percentile bootstrap for the mean final return (Confidence Bounds section).

Part 1 is small enough to check by hand. Part 2 draws many resamples of 5 and
10 seeds and measures how often the 95 percent interval covers the true mean.
The returns are made up.

Run: python3 bootstrap_example.py
"""

import numpy as np


def bootstrap_ci(x, idx, alpha=0.05):
    means = np.asarray(x, dtype=float)[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def main():
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    idx = np.array([[0, 0, 1, 1, 2], [3, 4, 4, 2, 1], [0, 1, 2, 3, 4], [4, 4, 4, 3, 3]])
    means = x[idx].mean(axis=1)
    lo, hi = bootstrap_ci(x, idx, alpha=0.5)
    print("x =", x.tolist(), " sample mean =", x.mean())
    print("resample means:", means.tolist())
    print(f"central 50% interval (25th and 75th percentile): ({lo}, {hi})")

    rng = np.random.default_rng(0)
    true_mean, sd = 3000.0, 800.0
    trials = 2000
    for n in (5, 10, 30):
        hit = 0
        width = 0.0
        for _ in range(trials):
            sample = rng.normal(true_mean, sd, size=n)
            idx = rng.integers(0, n, size=(1000, n))
            lo, hi = bootstrap_ci(sample, idx)
            hit += int(lo <= true_mean <= hi)
            width += hi - lo
        print(f"n={n:2d} seeds: 95% interval covers the true mean {hit / trials:.4f} of the time, mean width {width / trials:.1f}")


if __name__ == "__main__":
    main()
