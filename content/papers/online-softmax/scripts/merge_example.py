"""Merge two block summaries with the operator of Equation 4, and check its algebra.

Run: python3 scripts/merge_example.py
"""

import math
import random


def stats(chunk):
    m = max(chunk)
    return m, sum(math.exp(v - m) for v in chunk)


def oplus(a, b):
    """Equation 4: [m_i, d_i] (+) [m_j, d_j]."""
    (mi, di), (mj, dj) = a, b
    m = max(mi, mj)
    return m, di * math.exp(mi - m) + dj * math.exp(mj - m)


x = [2.0, 1.0, 4.0, 3.0]
left, right = stats(x[:2]), stats(x[2:])
print(f"left  block [2, 1]: m = {left[0]}, d = {left[1]:.4f}")
print(f"right block [4, 3]: m = {right[0]}, d = {right[1]:.4f}")
print(f"left (+) right    : m = {oplus(left, right)[0]}, d = {oplus(left, right)[1]:.4f}")
print(f"whole vector      : m = {stats(x)[0]}, d = {stats(x)[1]:.4f}")
print(f"left scaled by e^(2-4) = {math.exp(-2):.4f}: {left[1] * math.exp(-2):.4f}")

random.seed(1)


def rand_state():
    return random.gauss(0, 3), random.uniform(1, 5)


worst_assoc = worst_comm = 0.0
for _ in range(1000):
    a, b, c = rand_state(), rand_state(), rand_state()
    l, r = oplus(oplus(a, b), c), oplus(a, oplus(b, c))
    worst_assoc = max(worst_assoc, abs(l[1] - r[1]) / r[1])
    p, q = oplus(a, b), oplus(b, a)
    worst_comm = max(worst_comm, abs(p[1] - q[1]) / q[1])
print(f"largest relative gap, associativity: {worst_assoc:.1e}")
print(f"largest relative gap, commutativity: {worst_comm:.1e}")

# Blocks of 3 over 10 values, reduced left to right and as a tree.
y = [random.gauss(0, 4) for _ in range(10)]
blocks = [stats(y[i:i + 3]) for i in range(0, 10, 3)]
seq = blocks[0]
for b in blocks[1:]:
    seq = oplus(seq, b)
tree = oplus(oplus(blocks[0], blocks[1]), oplus(blocks[2], blocks[3]))
whole = stats(y)
print(f"10 values, 4 blocks (3+3+3+1): sequential d = {seq[1]:.6f}, tree d = {tree[1]:.6f}, whole d = {whole[1]:.6f}")
