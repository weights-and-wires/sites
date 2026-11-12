"""Split 10 seeds of one algorithm into two groups of 5 and compare them.

Both groups come from the same distribution, so every "significant" gap is a
false alarm. Compares the mean +/- 1 standard error band rule that the paper's
plots use with Welch's t-test at the 5 percent level. The two distributions
are made-up models of a final return, not data from the paper.

Run: python3 seed_split_sim.py
"""

import numpy as np

from stats_lib import t_two_sided_p, welch_t

E = 10000
N = 5


def draw(kind, rng):
    if kind == "normal":
        return rng.normal(3000.0, 800.0, size=(E, 2 * N))
    # Bimodal: each seed lands in a weak or a strong basin with equal chance.
    basin = rng.random(size=(E, 2 * N)) < 0.5
    return np.where(basin, 4000.0, 1500.0) + rng.normal(0.0, 300.0, size=(E, 2 * N))


def band_gap_rate(runs):
    a, b = runs[:, :N], runs[:, N:]
    se_a = a.std(axis=1, ddof=1) / np.sqrt(N)
    se_b = b.std(axis=1, ddof=1) / np.sqrt(N)
    return float(np.mean(np.abs(a.mean(axis=1) - b.mean(axis=1)) > se_a + se_b))


def welch_rate(runs, alpha=0.05):
    a, b = runs[:, :N], runs[:, N:]
    va = a.var(axis=1, ddof=1) / N
    vb = b.var(axis=1, ddof=1) / N
    t = (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (N - 1) + vb**2 / (N - 1))
    p = np.array([t_two_sided_p(ti, di) for ti, di in zip(t, df)])
    return float(np.mean(p < alpha))


def all5_better_rate(runs, alpha=0.05):
    """Group A reports all 5 seeds; the one-sided 'A is better' rate under the same rule."""
    hits = 0
    for row in runs:
        t, df = welch_t(row[:N], row[N:])
        hits += int(t > 0 and t_two_sided_p(t, df) < alpha)
    return hits / len(runs)


def top3_rate(runs, alpha=0.05):
    """Group A reports its best 3 of 5 seeds, group B reports all 5."""
    hits = 0
    for row in runs:
        a = np.sort(row[:N])[-3:]
        t, df = welch_t(a, row[N:])
        hits += int(t > 0 and t_two_sided_p(t, df) < alpha)
    return hits / len(runs)


def top3_mean_gain(runs):
    a = np.sort(runs[:, :N], axis=1)[:, -3:]
    return float(np.mean(a.mean(axis=1) - runs[:, N:].mean(axis=1)))


def main():
    rng = np.random.default_rng(0)
    print(f"{E} experiments, two groups of {N} seeds each, same algorithm")
    for kind in ("normal", "bimodal"):
        runs = draw(kind, rng)
        print(f"{kind:8s} band rule says 'different': {band_gap_rate(runs):.4f}   Welch p<0.05: {welch_rate(runs):.4f}")
        print(f"{'':8s} all 5 vs all 5, A 'significantly better': {all5_better_rate(runs):.4f}; best-3-of-5 vs all 5: {top3_rate(runs):.4f}, mean gain of A: {top3_mean_gain(runs):.1f}")


if __name__ == "__main__":
    main()
