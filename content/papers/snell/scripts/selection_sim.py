"""Simulate majority vote, best-of-N, and best-of-N weighted as N grows.

This is a toy model of this page, not the paper's experiment. Each sample is
correct with probability P_CORRECT. A wrong sample takes the popular wrong
answer with probability POPULAR, and otherwise one of 20 scattered wrong
answers. The verifier returns sigmoid(GAIN * (correct - 0.5) + SIGMA * z) with
z standard normal, so a larger SIGMA is a noisier verifier.

Run: python3 scripts/selection_sim.py
"""

import numpy as np

P_CORRECT = 0.25
POPULAR = 0.3
GAIN = 2.0
SIGMA = 1.5
TRIALS = 4000
BUDGETS = [1, 4, 16, 64, 256]
rng = np.random.default_rng(0)


def trial_batch(n):
    correct = rng.random((TRIALS, n)) < P_CORRECT
    popular = rng.random((TRIALS, n)) < POPULAR
    scattered = rng.integers(2, 22, size=(TRIALS, n))
    answers = np.where(correct, 0, np.where(popular, 1, scattered))
    z = rng.standard_normal((TRIALS, n))
    scores = 1.0 / (1.0 + np.exp(-(GAIN * (correct - 0.5) + SIGMA * z)))
    return answers, scores


def accuracy(n):
    answers, scores = trial_batch(n)
    best = answers[np.arange(TRIALS), scores.argmax(axis=1)]
    bon = float((best == 0).mean())
    maj_ok = 0
    wtd_ok = 0
    for a, s in zip(answers, scores):
        counts = np.bincount(a, minlength=22)
        maj_ok += int(counts.argmax() == 0)
        totals = np.bincount(a, weights=s, minlength=22)
        wtd_ok += int(totals.argmax() == 0)
    return bon, maj_ok / TRIALS, wtd_ok / TRIALS


print(f"p={P_CORRECT}, popular wrong share={POPULAR}, sigma={SIGMA}, trials={TRIALS}, seed 0")
print("   N   best-of-N  majority  weighted")
for n in BUDGETS:
    bon, maj, wtd = accuracy(n)
    print(f"{n:4d}   {bon:8.3f}  {maj:8.3f}  {wtd:8.3f}")
