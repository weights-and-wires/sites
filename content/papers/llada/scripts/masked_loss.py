"""Compute the numbers for the masked cross-entropy examples.

Run: python3 scripts/masked_loss.py
"""

import math

import numpy as np

# Hand example: L = 4 tokens, positions 1 and 3 are masked, t = 0.5.
# The model gives the true token probability 0.5 at position 1 and 0.25 at position 3.
t = 0.5
p_true = [0.5, 0.25]
nll = -sum(math.log(p) for p in p_true)
print(f"sum of -log p over masked positions = {nll:.4f}")
print(f"Equation 3 value, (1/t) * sum        = {nll / t:.4f}")
print(f"Algorithm 1 value, divided by L = 4  = {nll / t / 4:.4f}")
print(f"6 ln 2                               = {6 * math.log(2):.4f}")

# Why 1/t: a uniform model over V tokens has NLL exactly L ln V for every sequence.
# Given t, the number of masked tokens k is Binomial(L, t).
L, V = 8, 16
print(f"\nL ln V = {L * math.log(V):.4f}")
rng = np.random.default_rng(0)
n = 400_000
for t in (0.1, 0.5, 0.9):
    k = (rng.random((n, L)) < t).sum(axis=1)
    unweighted = (k * math.log(V)).mean()
    weighted = (k / t * math.log(V)).mean()
    print(f"t={t:3.1f}  mean loss without 1/t={unweighted:.4f} (exact {L * t * math.log(V):.4f})"
          f"  with 1/t={weighted:.4f}")
