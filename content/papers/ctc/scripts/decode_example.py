"""Compute the numbers for best path decoding and its failure case.

Run: python3 scripts/decode_example.py
"""

import itertools

import numpy as np

from ctc_ref import best_path, brute_force, collapse

np.set_printoptions(precision=4, suppress=True)
NAMES = {0: "-", 1: "a", 2: "b"}


def show(label):
    return "".join(NAMES[k] for k in label) or "(empty)"


print("Best path decoding on the page's T=3 example")
y = np.array([[0.4, 0.5, 0.1], [0.3, 0.4, 0.3], [0.5, 0.2, 0.3]])
pi = y.argmax(axis=1)
print("  argmax per frame =", "".join(NAMES[k] for k in pi), "-> B =", show(collapse(pi)))
print(f"  probability of that path = {np.prod(y.max(axis=1)):.4f}")
by_label = {}
for path in itertools.product(range(3), repeat=3):
    by_label.setdefault(collapse(path), brute_force(y, collapse(path))[0])
top = sorted(by_label.items(), key=lambda kv: -kv[1])[:3]
print("  top labellings:", [(show(k), round(v, 4)) for k, v in top])

print()
print("Where best path fails: T=2, blank 0.6, a 0.4 in both frames")
y2 = np.array([[0.6, 0.4, 0.0], [0.6, 0.4, 0.0]])
print("  best path =", "".join(NAMES[k] for k in y2.argmax(axis=1)), "-> B =", show(best_path(y2)),
      f"with path probability {0.6 * 0.6:.2f}")
for label in [(), (1,)]:
    p, hits = brute_force(y2, label)
    print(f"  p({show(label)}|x) = {p:.2f}  paths:", ", ".join(f"{''.join(NAMES[k] for k in h)}={q:.2f}" for h, q in hits))
print("  the best labelling is 'a' (0.64), but best path decoding returns the empty labelling (0.36)")

print()
print("Repeats merge first, and then blanks drop out")
for path in [(1, 1, 0, 1), (1, 0, 0, 1), (1, 1, 1, 1)]:
    print(f"  B({''.join(NAMES[k] for k in path)}) = {show(collapse(path))}")
