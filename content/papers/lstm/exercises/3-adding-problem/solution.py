import numpy as np


def make_adding_sequence(rng, T):
    """Return (X, target) for one adding-problem sequence (Section 5.4.1).

    X has shape (L, 2) with L drawn uniformly from T..T + T // 10. Column 0 holds
    values from [-1, 1]. Column 1 holds markers: 1.0 for the two marked pairs,
    -1.0 for the first and last pairs, and 0.0 elsewhere.
    """
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
    return X, float(target)


def cec_adder(X):
    """Add the marked values with an idealized memory cell and return the target."""
    s = 0.0
    for value, marker in X:
        input_gate = 1.0 if marker == 1.0 else 0.0
        s = s + input_gate * value
    return 0.5 + s / 4.0
