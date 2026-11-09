"""Compute the numbers for the RoPE-inside-LLaMA example.

Run: python3 scripts/rope_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def rope_adjacent(x, m, base=10000.0):
    """Rotate pairs (x[0], x[1]), (x[2], x[3]), ... by m * theta_j."""
    d = x.shape[-1]
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    c, s = np.cos(m * theta), np.sin(m * theta)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out


def rope_half(x, m, base=10000.0):
    """Rotate pairs (x[j], x[j + d/2]) by m * theta_j."""
    d = x.shape[-1]
    theta = base ** (-2.0 * np.arange(d // 2) / d)
    c, s = np.cos(m * theta), np.sin(m * theta)
    x1, x2 = x[: d // 2], x[d // 2 :]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c])


x = np.array([1.0, 2.0, 3.0, 4.0])
print("theta for d = 4:", (10000.0 ** (-2.0 * np.arange(2) / 4)).tolist())
print("x =", x.tolist(), "at m = 1")
print("adjacent pairs:", rope_adjacent(x, 1).tolist())
print("split halves:  ", rope_half(x, 1).tolist())

rng = np.random.default_rng(0)
q, k = rng.normal(size=8), rng.normal(size=8)
print()
print("score q_m . k_n for d = 8 (seed 0):")
for m, n in [(0, 0), (5, 5), (40, 40), (3, 1), (10, 8), (50, 48), (1, 4), (100, 103)]:
    print(f"  m = {m:3d}, n = {n:3d}, m - n = {m - n:2d}: {rope_adjacent(q, m) @ rope_adjacent(k, n):.6f}")
print("same pairs with split halves (relative property also holds):")
for m, n in [(3, 1), (10, 8), (50, 48)]:
    print(f"  m = {m:3d}, n = {n:3d}, m - n = {m - n:2d}: {rope_half(q, m) @ rope_half(k, n):.6f}")
print("no rotation: q . k =", round(float(q @ k), 6))
print("adjacent vs half at the same score (3, 1):", round(float(rope_adjacent(q, 3) @ rope_adjacent(k, 1)), 6), "vs", round(float(rope_half(q, 3) @ rope_half(k, 1)), 6))
