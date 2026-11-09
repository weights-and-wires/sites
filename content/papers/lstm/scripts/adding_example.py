"""Compute the numbers for the adding-problem example (Section 5.4.1).

Run: python3 scripts/adding_example.py
"""

import numpy as np


def make_adding_sequence(rng, T):
    L = int(rng.integers(T, T + T // 10 + 1))
    X = np.zeros((L, 2))
    X[:, 0] = rng.uniform(-1.0, 1.0, size=L)
    X[0, 1] = -1.0
    X[-1, 1] = -1.0
    first = int(rng.integers(0, 10))
    candidates = [j for j in range(T // 2 - 1) if j != first]
    second = int(rng.choice(candidates))
    for j in (first, second):
        X[j, 1] = 1.0
        if j == 0:
            X[j, 0] = 0.0
    target = 0.5 + (X[first, 0] + X[second, 0]) / 4.0
    return X, float(target), first, second


def cec_adder(X):
    s = 0.0
    for value, marker in X:
        s = s + (1.0 if marker == 1.0 else 0.0) * value
    return 0.5 + s / 4.0


rng = np.random.default_rng(0)
X, target, first, second = make_adding_sequence(rng, 100)
print(f"length = {len(X)}, marked positions = {first} and {second}")
print(f"X1 = {X[first, 0]:.4f}, X2 = {X[second, 0]:.4f}, X1 + X2 = {X[first, 0] + X[second, 0]:.4f}")
print(f"target = 0.5 + (X1 + X2) / 4 = {target:.4f}")
print(f"idealized CEC output = {cec_adder(X):.4f}")

# Baseline: always answer 0.5, the mean target. A sequence counts as correct when the
# absolute error is below 0.04 (Section 5.4.1).
rng = np.random.default_rng(1)
n_seq = 2560
targets, cec_errors = [], 0
for _ in range(n_seq):
    X, target, _, _ = make_adding_sequence(rng, 100)
    targets.append(target)
    cec_errors += abs(cec_adder(X) - target) >= 0.04
targets = np.array(targets)
wrong = int(np.sum(np.abs(0.5 - targets) >= 0.04))
print(f"constant 0.5: mean |error| = {np.mean(np.abs(0.5 - targets)):.4f}; wrong on {wrong} of {n_seq}")
print(f"idealized CEC: wrong on {cec_errors} of {n_seq}")
