"""Forward and backward variables, the diagonal identity, and the gradient
(Section 2.4 and Section 2.5, Equations 16 to 22).

Uses the same toy as lattice_example.py.

Run: python3 scripts/forward_example.py
"""

import numpy as np

from rnnt_ref import (backward, brute_force, count_alignments, emit_null, forward,
                      gradients, lattice_probs, neg_log_prob)

np.set_printoptions(precision=4, suppress=True, linewidth=120)

F = np.array([[1, 3, 1], [1, 1, 3], [2, 1, 1]], dtype=float)
G = np.array([[1, 1, 1], [1, 1, 2], [4, 1, 1]], dtype=float)
f, g = np.log(F), np.log(G)
y = (1, 2)
T, U = 3, len(y)

emit, null = emit_null(lattice_probs(f, g), y)
alpha, total = forward(emit, null)
print("alpha(t, u), rows t = 1..3, columns u = 0..2")
print(alpha)
print(f"Pr(y|x) = alpha(T, U) * null(T, U) = {alpha[-1, -1]:.4f} * {null[-1, -1]:.1f} = {total:.4f}")
print("alpha(2, 1) = alpha(1, 1) null(1, 1) + alpha(2, 0) y(2, 0) =",
      f"{alpha[0, 1]:.4f} * {null[0, 1]:.4f} + {alpha[1, 0]:.4f} * {emit[1, 0]:.4f} = {alpha[1, 1]:.4f}")

bf_total, paths = brute_force(emit, null)
print(f"\nBrute force over {len(paths)} alignments (formula C(T+U-1, U) = {count_alignments(T, U)}):")
for s, p in paths:
    print(f"  {s}  {p:.5f}")
print(f"  sum = {bf_total:.4f}")

beta = backward(emit, null)
print("\nbeta(t, u), rows t = 1..3, columns u = 0..2")
print(beta)
print("alpha * beta")
print(alpha * beta)
print("Equation 19: sum of alpha * beta over each diagonal t + u = n")
for n in range(1, T + U + 1):
    s = sum(alpha[t - 1, n - t] * beta[t - 1, n - t] for t in range(1, T + 1) if 0 <= n - t <= U)
    print(f"  n={n}: {s:.4f}")

dfdL, dgdL, _ = gradients(f, g, y)
print("\ndL/df (rows t = 1..3, columns null, a, b), Equations 20 and 21")
print(dfdL)
print("dL/dg (rows u = 0..2), Equations 20 and 22")
print(dgdL)
print("Row sums of dL/df:", dfdL.sum(axis=1), " (each is 0 because the softmax is shift invariant)")

eps = 1e-6
worst = 0.0
for arr, grad in ((f, dfdL), (g, dgdL)):
    for idx in np.ndindex(*arr.shape):
        orig = arr[idx]
        arr[idx] = orig + eps
        up = neg_log_prob(f, g, y)
        arr[idx] = orig - eps
        down = neg_log_prob(f, g, y)
        arr[idx] = orig
        worst = max(worst, abs((up - down) / (2 * eps) - grad[idx]))
print(f"Largest |analytic - finite difference| over all entries of f and g: {worst:.1e}")
