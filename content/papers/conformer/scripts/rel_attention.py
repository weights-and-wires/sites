#!/usr/bin/env python3
"""Numbers for the relative position idea (Section 2.1, Transformer-XL scores)."""

import numpy as np


def rel_scores(q, k, p, u, v):
    T, dk = q.shape
    content = (q + u) @ k.T
    pos = (q + v) @ p.T
    idx = (T - 1) - np.arange(T)[:, None] + np.arange(T)[None, :]
    return (content + np.take_along_axis(pos, idx, axis=1)) / np.sqrt(dk)


def naive(q, k, p, u, v):
    T, dk = q.shape
    A = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            A[i, j] = (np.dot(q[i] + u, k[j]) + np.dot(q[i] + v, p[(T - 1) - (i - j)])) / np.sqrt(dk)
    return A


q = np.array([[1.0], [2.0]])
k = np.array([[3.0], [4.0]])
p = np.array([[10.0], [20.0], [30.0]])
u = np.array([1.0])
v = np.array([2.0])
print("scores:", rel_scores(q, k, p, u, v).tolist())

rng = np.random.default_rng(0)
T, dk = 6, 4
q, k = rng.normal(size=(T, dk)), rng.normal(size=(T, dk))
p = rng.normal(size=(2 * T - 1, dk))
u, v = rng.normal(size=dk), rng.normal(size=dk)
print("max |vectorized - naive|: %.2e" % np.abs(rel_scores(q, k, p, u, v) - naive(q, k, p, u, v)).max())

# Shift invariance of the position term.
T, dk = 5, 3
q = np.tile(rng.normal(size=(1, dk)), (T, 1))
k = np.zeros((T, dk))
p = rng.normal(size=(2 * T - 1, dk))
A = rel_scores(q, k, p, np.zeros(dk), np.zeros(dk))
print("position term shift-invariant, max |A[i,j] - A[i+1,j+1]|: %.1e" % max(abs(A[i, j] - A[i + 1, j + 1]) for i in range(T - 1) for j in range(T - 1)))

# Table size: relative offsets versus absolute positions.
for L in (100, 1000):
    print(f"T={L}: {2 * L - 1} relative offsets, {L * L} (i, j) pairs")
