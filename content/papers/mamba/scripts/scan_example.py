"""Compute the numbers for the parallel scan example (Section 3.3.2).

The recurrence h_t = a_t h_{t-1} + b_t is a pair (a_t, b_t) under an
associative operator. This script runs a sequential loop and a log-depth
Hillis-Steele scan on the same pairs and prints the intermediate rounds.

Run: python3 scripts/scan_example.py
"""

import math

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def combine(left, right):
    a1, b1 = left
    a2, b2 = right
    return a2 * a1, a2 * b1 + b2


def sequential(a, b):
    h, out = 0.0, []
    for at, bt in zip(a, b):
        h = at * h + bt
        out.append(h)
    return np.array(out)


def hillis_steele(a, b, verbose=False):
    a, b = np.array(a, float), np.array(b, float)
    L = len(a)
    d, rnd = 1, 0
    while d < L:
        na, nb = combine((a[:-d], b[:-d]), (a[d:], b[d:]))
        a = np.concatenate([a[:d], na])
        b = np.concatenate([b[:d], nb])
        rnd += 1
        if verbose:
            print(f"round {rnd} (offset {d}): a = {a.tolist()}  b = {b.tolist()}")
        d *= 2
    return b


# The hand example: a_t = 0.5, b_t = B_bar * x_t = 0.5 * x_t with x = 1, 2, 3, 4.
a = [0.5, 0.5, 0.5, 0.5]
b = [0.5, 1.0, 1.5, 2.0]
print("pairs (a_t, b_t):", list(zip(a, b)))
print("sequential h_t  :", sequential(a, b).tolist())
print("scan h_t        :", hillis_steele(a, b, verbose=True).tolist())

# Non-commutativity: the operator needs its order.
p, q = (0.5, 1.0), (0.25, 2.0)
print()
print("combine(p, q) =", combine(p, q))
print("combine(q, p) =", combine(q, p))

# Associativity on random pairs.
rng = np.random.default_rng(0)
x, y, z = [tuple(rng.normal(size=2)) for _ in range(3)]
lhs = combine(combine(x, y), z)
rhs = combine(x, combine(y, z))
print("associative on random pairs:", np.allclose(lhs, rhs))

# Agreement and depth for several lengths.
print()
print("L     rounds  max |scan - sequential|")
for L in [1, 2, 7, 8, 100, 1000]:
    aa = rng.uniform(0.5, 1.0, size=L)
    bb = rng.normal(size=L)
    err = np.abs(hillis_steele(aa, bb) - sequential(aa, bb)).max()
    rounds = math.ceil(math.log2(L)) if L > 1 else 0
    print(f"{L:<5} {rounds:<7} {err:.2e}")

# Work: the sequential loop does L - 1 combines in L - 1 dependent steps.
# Hillis-Steele does about L log2 L combines in ceil(log2 L) rounds.
print()
L = 1024
work = sum(L - 2**k for k in range(int(math.log2(L))))
print(f"L = {L}: sequential = {L - 1} dependent steps; Hillis-Steele = {int(math.log2(L))} rounds, {work} combines")
