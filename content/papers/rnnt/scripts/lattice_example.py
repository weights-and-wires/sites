"""The output distribution over the lattice (Section 2.3, Equations 12 to 15).

The weights are stored as logs, so exp(f + g) is a product of small integers
that you can check by hand.

Run: python3 scripts/lattice_example.py
"""

import numpy as np

from rnnt_ref import emit_null, lattice_probs

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# Columns: null, a, b. Three frames (T = 3); target y = (a, b), so U = 2.
F = np.array([[1, 3, 1], [1, 1, 3], [2, 1, 1]], dtype=float)
# Three prediction vectors: g_0, g_1, g_2 (U + 1 = 3).
G = np.array([[1, 1, 1], [1, 1, 2], [4, 1, 1]], dtype=float)
f, g = np.log(F), np.log(G)
y = (1, 2)

print("exp(f_t) rows t = 1..3 (columns: null, a, b)")
print(F)
print("exp(g_u) rows u = 0..2")
print(G)

probs = lattice_probs(f, g)
emit, null = emit_null(probs, y)
print("\nPr(k | t, u) at (t=2, u=1): h = exp(f_2) * exp(g_1) =", F[1] * G[1], " sum =", (F[1] * G[1]).sum())
print("  probabilities =", probs[1, 1], " sum =", round(probs[1, 1].sum(), 10))

print("\nnull(t, u), rows t = 1..3, columns u = 0..2")
print(null)
print("y(t, u) = Pr(y_{u+1} | t, u), rows t = 1..3, columns u = 0..1")
print(emit)
print("\nEvery node sums to 1:", np.allclose(probs.sum(axis=-1), 1.0))
print("Number of nodes (t, u):", probs.shape[0] * probs.shape[1], " softmax size K + 1 =", probs.shape[2])
