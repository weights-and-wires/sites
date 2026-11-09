"""Compute the numbers for the collapse map and the sum over alignments.

Run: python3 scripts/collapse_example.py
"""

import numpy as np

from ctc_ref import BLANK, brute_force, collapse

np.set_printoptions(precision=4, suppress=True)
NAMES = {0: "-", 1: "a", 2: "b"}


def show(path):
    return "".join(NAMES[k] for k in path)


print("Collapse map B (blank is '-')")
for p in ["a-ab-", "-aa--abb", "aab", "a-a", "aa", "---"]:
    ids = [{"-": 0, "a": 1, "b": 2}[c] for c in p]
    print(f"  B({p}) = {show(collapse(ids)) or '(empty)'}")

# Posterior matrix y[t, k]: columns are blank, a, b. Each row sums to 1.
y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])
print()
print("y (T=3; columns: blank, a, b)")
print(y)
print("row sums =", y.sum(axis=1).tolist())

print()
for label in [(1, 2), (1, 1), (1,), ()]:
    total, hits = brute_force(y, label)
    name = show(label) or "(empty)"
    print(f"p({name}|x) = {total:.4f}  from {len(hits)} paths")
    for path, p in hits:
        print(f"    {show(path)}  {p:.4f}")

print()
import itertools

by_label = {}
for path in itertools.product(range(3), repeat=3):
    by_label[collapse(path)] = by_label.get(collapse(path), 0.0) + float(np.prod([y[t, k] for t, k in enumerate(path)]))
print(f"{len(by_label)} distinct labellings from 27 paths; their probabilities add to {sum(by_label.values()):.4f}")

print()
import math

print("Segmentations for framewise training: split T frames into U non-empty consecutive segments")
for T, U in [(100, 3), (300, 30)]:
    print(f"  T={T}, U={U}: C(T-1, U-1) = {math.comb(T - 1, U - 1):.4g}")

print()
print("Minimum frames for a label: |l| plus one frame per adjacent repeated pair")
for word in ["cat", "aa", "hello", "aaa"]:
    need = len(word) + sum(1 for i in range(1, len(word)) if word[i] == word[i - 1])
    print(f"  {word}: {need}")
