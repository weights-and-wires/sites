"""Best-of-N accuracy against N for a noisy scorer, and majority voting as a baseline.

Model (this page's, not the paper's): each sampled solution is correct with
probability p. A scorer returns sep * correct + sigma * N(0, 1). Wrong solutions
carry one of W = 20 equally likely wrong answers, which majority voting counts.

Run: python3 scripts/best_of_n_sim.py
"""

from collections import Counter

import numpy as np

P, W, TRIALS = 0.30, 20, 4000
NS = [1, 2, 4, 8, 16, 64, 256]


def best_of_n(n, sep, sigma, rng):
    correct = rng.random((TRIALS, n)) < P
    scores = sep * correct + sigma * rng.standard_normal((TRIALS, n))
    pick = scores.argmax(axis=1)
    return correct[np.arange(TRIALS), pick].mean()


def majority(n, rng):
    correct = rng.random((TRIALS, n)) < P
    wrong = 1 + rng.integers(0, W, (TRIALS, n))
    answers = np.where(correct, 0, wrong)
    hits = 0
    for row in answers.tolist():
        hits += Counter(row).most_common(1)[0][0] == 0
    return hits / TRIALS


rng = np.random.default_rng(0)
print(f"p = {P}, wrong answers spread over {W} values, {TRIALS} trials per cell")
print(f"{'N':>4} {'oracle':>7} {'sigma=0.5':>10} {'sigma=1':>8} {'sigma=2':>8} {'random':>7} {'majority':>9}  1-(1-p)^N")
for n in NS:
    oracle = best_of_n(n, 1.0, 0.0, rng)
    s05 = best_of_n(n, 1.0, 0.5, rng)
    s1 = best_of_n(n, 1.0, 1.0, rng)
    s2 = best_of_n(n, 1.0, 2.0, rng)
    rand = best_of_n(n, 0.0, 1.0, rng)
    maj = majority(n, rng)
    print(f"{n:4d} {oracle:7.3f} {s05:10.3f} {s1:8.3f} {s2:8.3f} {rand:7.3f} {maj:9.3f}  {1 - (1 - P) ** n:.3f}")
