"""Bootstrap power analysis with a uniform lift (Power Analysis section).

Take a sample of final returns, scale a copy of it by a lift (1.25 in the
paper), and bootstrap both. The power is the fraction of resample pairs where
Welch's t-test calls the lifted copy different. The samples are made up: one
stable (like TRPO on Hopper in the paper) and one noisy.

Run: python3 power_example.py
"""

import numpy as np

from stats_lib import t_critical, welch_t_rows


def lift_power(x, lift, idx_a, idx_b, crit):
    x = np.asarray(x, dtype=float)
    a = x[idx_a]
    b = lift * x[idx_b]
    return float(np.mean(np.abs(welch_t_rows(a, b)) > crit))


def main():
    samples = {
        "stable": np.array([2900.0, 3000.0, 3100.0, 2950.0, 3050.0]),
        "noisy": np.array([800.0, 4200.0, 2500.0, 5200.0, 1300.0]),
    }
    rng = np.random.default_rng(0)
    B = 2000
    for name, x in samples.items():
        cv = x.std(ddof=1) / x.mean()
        print(f"{name}: mean {x.mean():.1f}, sd {x.std(ddof=1):.1f}, coefficient of variation {cv:.3f}")
        for n in (5, 10, 20, 40):
            idx_a = rng.integers(0, len(x), size=(B, n))
            idx_b = rng.integers(0, len(x), size=(B, n))
            crit = t_critical(2 * n - 2)
            print(f"  {n:2d} seeds per group, lift 1.25: power {lift_power(x, 1.25, idx_a, idx_b, crit):.3f}")


if __name__ == "__main__":
    main()
