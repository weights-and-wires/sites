"""Compute the numbers for the "why divide by sqrt(d_k)" example.

Run: python3 scripts/scaling_variance.py
"""

import numpy as np

np.set_printoptions(precision=5, suppress=True)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


rng = np.random.default_rng(0)
d_k, n = 64, 100_000
q = rng.standard_normal((n, d_k))
k = rng.standard_normal((n, d_k))
dots = (q * k).sum(axis=1)
print(f"d_k = {d_k}, samples = {n}")
print(f"variance of q.k          = {dots.var():.2f}   (theory: {d_k})")
print(f"variance of q.k/sqrt(d_k) = {(dots / np.sqrt(d_k)).var():.2f}   (theory: 1)")
print()
print("softmax([8, 0]) =", softmax(np.array([8.0, 0.0])).tolist())
print("softmax([1, 0]) =", softmax(np.array([1.0, 0.0])).tolist())

# Gradient of the largest weight with respect to its score is p (1 - p).
for gap in (8.0, 1.0):
    p = softmax(np.array([gap, 0.0]))[0]
    print(f"gap {gap:.0f}: p = {p:.5f}, dp/ds = p(1-p) = {p * (1 - p):.5f}")

# Typical spread of a score row at d_k = 64 with n = 10 keys.
scores = rng.standard_normal((10,)) * np.sqrt(d_k)
print()
print("a row of 10 unscaled scores (d_k = 64): max weight =", round(float(softmax(scores).max()), 4))
print("the same row after scaling:            max weight =", round(float(softmax(scores / np.sqrt(d_k)).max()), 4))
