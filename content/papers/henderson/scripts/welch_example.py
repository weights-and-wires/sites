"""Welch's t-test by hand (Significance section) and the pooled version.

Run: python3 welch_example.py
"""

import numpy as np

from stats_lib import t_two_sided_p, welch_t


def main():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    va, vb = a.var(ddof=1), b.var(ddof=1)
    print(f"means {a.mean()} and {b.mean()}, sample variances {va} and {vb}")
    se2 = va / len(a) + vb / len(b)
    print(f"var(a)/n + var(b)/n = {se2}")
    t, df = welch_t(a, b)
    print(f"Welch t = {t:.4f}, df = {df:.4f}, two-sided p = {t_two_sided_p(t, df):.4f}")
    sp2 = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    t_pool = (a.mean() - b.mean()) / np.sqrt(sp2 * (1 / len(a) + 1 / len(b)))
    print(f"pooled t = {t_pool:.4f}, df = 8, two-sided p = {t_two_sided_p(t_pool, 8):.4f}")
    # Same data: Welch with equal group sizes and equal variances gives df = 2n - 2.
    c = a + 10.0
    t2, df2 = welch_t(a, c)
    print(f"equal variances: df = {df2:.4f}")


if __name__ == "__main__":
    main()
