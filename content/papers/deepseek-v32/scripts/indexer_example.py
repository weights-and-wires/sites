"""Lightning indexer scores and top-k selection on a toy sequence.

Equation 1 (Section 2.1): I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s]).
Equation 2 keeps the keys with the k largest scores among s <= t.

The paper does not give H^I or d^I, so this toy uses H^I = 2 and d^I = 2.

Run: python3 scripts/indexer_example.py
"""

import numpy as np

L, H, d, k = 6, 2, 2, 3
K = np.array([[1, 0], [0, 1], [1, 1], [-1, 1], [1, -1], [2, 0]], dtype=float)
Q = np.array(
    [
        [[1, 0], [0, 1]],
        [[1, 1], [-1, 0]],
        [[0, 1], [1, -1]],
        [[1, 0], [1, 1]],
        [[-1, 1], [2, 1]],
        [[1, 1], [0, -1]],
    ],
    dtype=float,
)
W = np.array([[1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [0.5, 0.5], [1.0, 0.25], [0.25, 1.0]])

# raw[t, s] = sum_j W[t, j] * relu(Q[t, j] . K[s])
dots = np.einsum("tjd,sd->tjs", Q, K)
raw = np.einsum("tj,tjs->ts", W, np.maximum(dots, 0))
I = np.where(np.tril(np.ones((L, L), dtype=bool)), raw, -np.inf)
print("I (future positions are -inf):")
for t in range(L):
    print(t, [("-inf" if np.isinf(x) else float(x)) for x in I[t]])

t, s = 3, 2
print(f"\nworked cell t={t}, s={s}")
print("q[t,0], q[t,1] =", Q[t, 0], Q[t, 1], " k[s] =", K[s], " w[t] =", W[t])
print("dot products   =", dots[t, 0, s], dots[t, 1, s])
print("I[t,s]         =", raw[t, s])

print(f"\ntop-{k} selection per query")
for t in range(L):
    n = min(k, t + 1)
    sel = sorted(np.argsort(-I[t], kind="stable")[:n].tolist())
    print(f"t={t}: selected {n} of {t + 1} visible keys ->", sel)

# ReLU effect: negative dot products give zero, not a negative score.
neg = np.minimum(dots, 0).sum()
print("\nsum of negative dot products clipped by ReLU =", float(neg))
