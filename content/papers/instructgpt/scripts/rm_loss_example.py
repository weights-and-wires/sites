"""Compute the numbers for the reward-model loss examples (Equation 1).

Run: python3 content/papers/instructgpt/scripts/rm_loss_example.py
"""

from itertools import combinations
from math import comb

import numpy as np

# One prompt, K = 4 completions A, B, C, D, with made-up reward-model scores.
names = ["A", "B", "C", "D"]
r = np.array([0.2, 1.5, -0.7, 0.9])
# The labeler's ranking, best first: B > D > A > C.
ranking = [1, 3, 0, 2]


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def pair_losses(r, ranking):
    """Return [(winner, loser, r_w - r_l, -log sigma(r_w - r_l))] over all C(K, 2) pairs."""
    out = []
    for i, j in combinations(range(len(ranking)), 2):
        w, l = ranking[i], ranking[j]
        d = r[w] - r[l]
        out.append((w, l, d, -log_sigmoid(d)))
    return out


rows = pair_losses(r, ranking)
print("pairs (winner > loser, r_w - r_l, -log sigmoid):")
for w, l, d, nll in rows:
    print(f"  {names[w]} > {names[l]}  diff={d:+.1f}  loss={nll:.4f}")
loss = np.mean([nll for *_, nll in rows])
print(f"number of pairs C(4,2) = {comb(4, 2)}")
print(f"Equation 1 loss (mean over pairs) = {loss:.4f}")
print(f"sum over pairs (not divided)      = {sum(nll for *_, nll in rows):.4f}")

# Shift invariance: add 100 to every score.
rows_shift = pair_losses(r + 100.0, ranking)
print(f"loss after adding 100 to every score = {np.mean([nll for *_, nll in rows_shift]):.4f}")

# A labeler ranking that disagrees with the scores: reverse it.
rows_rev = pair_losses(r, ranking[::-1])
print(f"loss with the ranking reversed = {np.mean([nll for *_, nll in rows_rev]):.4f}")

# Two completions: the loss is -log sigmoid(diff).
print(f"K = 2, diff = 1.0: loss = {-log_sigmoid(1.0):.4f}, P(winner preferred) = {np.exp(log_sigmoid(1.0)):.4f}")

# Cost of treating pairs as separate data points versus one batch element per prompt.
print("K  pairs  (the paper counts C(K,2) forward passes if pairs are separate, K if the prompt is one batch element)")
for K in range(4, 10):
    print(f"{K}  {comb(K, 2):>5}  separate: {comb(K, 2)} passes; one batch element: {K} passes")
print(f"largest batch: 64 prompts x C(9,2) = {64 * comb(9, 2)} comparisons (Appendix C.2)")
