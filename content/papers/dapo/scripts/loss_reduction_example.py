"""Compare the sample-level loss of GRPO with the token-level loss of DAPO
(Section 2.2, Equation 5; Section 3.3, Equation 12).

Run: python3 scripts/loss_reduction_example.py
"""

import numpy as np

# Two outputs for one question. The short one is good (A = +1, 2 tokens) and the
# long one is bad (A = -1, 8 tokens). Take r = 1, so each token term equals A.
lengths = [2, 8]
advantages = [1.0, -1.0]
terms = [np.full(n, a) for n, a in zip(lengths, advantages)]
G = len(lengths)
total = sum(lengths)

sample_level = np.mean([t.mean() for t in terms])
token_level = sum(t.sum() for t in terms) / total

print("Per-token weight in the objective")
for n in lengths:
    print(f"|o| = {n}: sample-level 1/(G*|o|) = {1 / (G * n):.4f}, token-level 1/sum|o| = {1 / total:.4f}")

print()
print(f"Sample-level objective (Equation 5):  {sample_level:+.4f}")
print(f"Token-level objective (Equation 12):  {token_level:+.4f}")

print()
print("Share of the total weight that each output's tokens hold")
for n in lengths:
    s = n * (1 / (G * n))
    t = n * (1 / total)
    print(f"|o| = {n}: sample-level {s:.2f}, token-level {t:.2f}")
