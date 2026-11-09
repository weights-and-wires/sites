"""Compute the numbers for the expected tokens per iteration (Section 3.1, Equation 1).

Run: python3 scripts/tokens_per_iteration.py
"""

import numpy as np


def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


alpha, gamma = 0.8, 5
print(f"alpha = {alpha}, gamma = {gamma}")
print("alpha^(gamma+1)          :", round(alpha ** (gamma + 1), 6))
print("E(#generated tokens)     :", round(expected_tokens(alpha, gamma), 4))
terms = [alpha**k for k in range(gamma + 1)]
print("sum_{k=0}^{gamma} alpha^k:", round(sum(terms), 4))

# Distribution of the tokens per iteration: capped geometric.
print()
print("P(n accepted) for n = 0..gamma-1 is alpha^n (1 - alpha); P(n = gamma) = alpha^gamma")
probs = [alpha**n * (1 - alpha) for n in range(gamma)] + [alpha**gamma]
print("probabilities            :", [round(x, 4) for x in probs], "sum", round(sum(probs), 4))
mean = sum((n + 1) * pr for n, pr in enumerate(probs))
print("mean of n + 1            :", round(mean, 4))

rng = np.random.default_rng(0)
trials = 200_000
u = rng.random((trials, gamma))
rejected = u > alpha  # position i rejected with probability 1 - alpha
n_acc = np.where(rejected.any(axis=1), rejected.argmax(axis=1), gamma)
print(f"seeded simulation, {trials} iterations: mean tokens", round(float((n_acc + 1).mean()), 3))

print()
print("expected tokens, rows alpha, columns gamma = 1, 3, 5, 7")
for a in (0.5, 0.7, 0.8, 0.9):
    print(f"alpha {a}:", [round(expected_tokens(a, g), 3) for g in (1, 3, 5, 7)], " limit 1/(1-alpha) =", round(1 / (1 - a), 3))
