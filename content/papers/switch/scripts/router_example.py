#!/usr/bin/env python3
"""Worked example for the page: top-1 routing, capacity, token dropping, and
the load-balancing loss on 6 tokens and 3 experts.

Notation follows the paper: h(x) = W_r x are the router logits (Section 2.1),
p_i(x) is the softmax over experts (Equation 1), f_i and P_i are the dispatch
and probability fractions (Equations 5 and 6), and the auxiliary loss is
alpha * N * sum_i f_i P_i (Equation 4).
"""

import math

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Router logits h(x) for T = 6 tokens and N = 3 experts. They are written
# down directly, so the example does not depend on a random number generator.
h = np.array([
    [2.0, 0.0, 0.0],
    [1.5, 0.5, 0.0],
    [1.0, 0.0, 0.5],
    [2.0, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 2.0],
])
T, N = h.shape
alpha = 1e-2

e = np.exp(h - h.max(axis=1, keepdims=True))
p = e / e.sum(axis=1, keepdims=True)  # Equation 1
idx = p.argmax(axis=1)  # top-1 choice
gate = p[np.arange(T), idx]  # the gate value that scales the expert output

print("router probabilities p_i(x):")
print(p)
print("chosen expert:", idx.tolist())
print("gate values p_i(x):", gate)

# Equation 3: expert capacity. The paper does not say how to round; this
# script rounds up.
for cf in (1.0, 1.5):
    capacity = math.ceil(T / N * cf)
    counts = np.bincount(idx, minlength=N)
    # Earlier tokens get the slots first.
    seen = np.zeros(N, dtype=int)
    kept = np.zeros(T, dtype=bool)
    for t in range(T):
        if seen[idx[t]] < capacity:
            kept[t] = True
        seen[idx[t]] += 1
    print(f"capacity factor {cf}: capacity {capacity}, load {counts.tolist()}, "
          f"dropped tokens {np.flatnonzero(~kept).tolist()} ({(~kept).mean():.3f} of the batch)")

# Equations 5 and 6.
f = np.bincount(idx, minlength=N) / T
P = p.mean(axis=0)
loss = alpha * N * np.sum(f * P)  # Equation 4
print("f =", f)
print("P =", P)
print("sum f_i P_i =", round(float(np.sum(f * P)), 4))
print("aux loss alpha*N*sum f_i P_i =", round(float(loss), 6))

# Reference points: perfectly uniform routing gives alpha; full collapse onto
# one expert gives at most alpha * N.
print("uniform routing: loss =", round(alpha * N * N * (1 / N) * (1 / N), 6), "= alpha")
print("collapse onto one expert (P_1 = 1): loss =", alpha * N * 1.0 * 1.0, "= alpha * N")

# Why f needs argmax and P needs probabilities: P alone is a soft summary.
print("share of the batch for expert 0: f_0 =", round(float(f[0]), 4), "but P_0 =", round(float(P[0]), 4))
