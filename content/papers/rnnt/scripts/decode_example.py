"""Greedy decoding, the beam search of Section 2.6, and the exact mode, on a toy.

The prediction network is a lookup table: its output depends only on the last
label emitted. The weights are exp(f_t) and exp(g_u), so every Pr(k | t, u)
is a ratio of integer products.

Run: python3 scripts/decode_example.py
"""

import itertools
import math

import numpy as np

from rnnt_ref import emit_null, forward, lattice_probs

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# Columns: null, a, b. Three frames. exp(f_t):
F = np.array([[4, 4, 4], [6, 2, 1], [4, 4, 2]], dtype=float)
# exp(g) after the start (key 0) and after an `a` (key 1) or a `b` (key 2).
G_TABLE = {0: [5, 4, 1], 1: [5, 1, 1], 2: [5, 1, 4]}
K = 2
NAMES = {1: "a", 2: "b"}


def g_weights(prefix):
    return np.array(G_TABLE[prefix[-1] if prefix else 0], dtype=float)


def node_probs(t, prefix):
    """Pr(k | t, u) for the node reached by emitting `prefix` during frames 1..t."""
    w = F[t] * g_weights(prefix)
    return w / w.sum()


def exact_prob(y):
    G = np.stack([g_weights(tuple(y[:u])) for u in range(len(y) + 1)])
    emit, null = emit_null(lattice_probs(np.log(F), np.log(G)), y)
    return forward(emit, null)[1]


def show(y):
    return "".join(NAMES[k] for k in y) or "(empty)"


def greedy(max_symbols=3):
    y = []
    for t in range(len(F)):
        for _ in range(max_symbols):
            k = int(np.argmax(node_probs(t, tuple(y))))
            if k == 0:
                break
            y.append(k)
    return tuple(y)


def beam_search(width):
    """Algorithm 1 of Section 2.6: expand the most probable hypothesis, send its
    null mass to the next frame, add one-label extensions to the same frame."""
    B = {(): 1.0}
    for t in range(len(F)):
        A, B = dict(B), {}
        while A:
            top = max(A.values())
            if sum(1 for v in B.values() if v > top) >= width:
                break
            y = max(A, key=A.get)
            p = A.pop(y)
            pr = node_probs(t, y)
            B[y] = B.get(y, 0.0) + p * pr[0]
            for k in range(1, K + 1):
                A[y + (k,)] = A.get(y + (k,), 0.0) + p * pr[k]
        B = dict(sorted(B.items(), key=lambda kv: -kv[1])[:width])
    best = max(B, key=lambda y: math.log(B[y]) / max(len(y), 1))
    return best, B


print("Pr(k | t, u) at the start (u = 0), rows t = 1..3, columns null, a, b")
print(np.stack([node_probs(t, ()) for t in range(3)]))

exact = {}
for L in range(0, 10):
    for y in itertools.product(range(1, K + 1), repeat=L):
        exact[y] = exact_prob(y)
    print(f"sum of Pr(y|x) over all y with |y| <= {L}: {sum(exact.values()):.4f}")

mode = max(exact, key=exact.get)
print("\nTop four sequences by Pr(y|x):")
for y in sorted(exact, key=exact.get, reverse=True)[:4]:
    print(f"  {show(y):8s} {exact[y]:.4f}")
print(f"Exact mode: {show(mode)} with Pr = {exact[mode]:.4f}")

g = greedy()
print(f"\nGreedy decoding (at most 3 labels per frame): {show(g)} with Pr = {exact[g]:.4f}")
for t in range(3):
    print(f"  frame {t + 1}: argmax over {node_probs(t, ()).round(4)} -> {['null', 'a', 'b'][int(np.argmax(node_probs(t, ())))]}")

for width in (1, 2, 4):
    best, B = beam_search(width)
    print(f"Beam search, width {width}: {show(best)}   hypotheses kept at the end:",
          {show(y): round(float(p), 4) for y, p in B.items()})
