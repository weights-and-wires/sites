#!/usr/bin/env python3
"""Check FlashAttention-2 Algorithm 2 (backward pass) and its column-block parallelism.

Each column block j produces its own dK_j and dV_j. They share only dQ, which
every column block adds into. The script runs the column blocks in a shuffled
order, as thread blocks with atomic adds would, and compares with the
gradients from the standard formulas of Section 2.2.

Run: python3 content/papers/fa2/scripts/backward_check.py
"""

import numpy as np

rng = np.random.default_rng(1)
N, d, Br, Bc = 12, 4, 4, 3
Q, K, V, dO = (rng.normal(size=(N, d)) for _ in range(4))

# Standard backward (Section 2.2).
S = Q @ K.T
P = np.exp(S - S.max(axis=1, keepdims=True))
P /= P.sum(axis=1, keepdims=True)
O = P @ V
dV_ref = P.T @ dO
dP = dO @ V.T
dS_ref = P * (dP - (P * dP).sum(axis=1, keepdims=True))
dQ_ref, dK_ref = dS_ref @ K, dS_ref.T @ Q

# Algorithm 2 with the logsumexp L and D = rowsum(dO * O).
L = np.log(np.exp(S).sum(axis=1))
D = (dO * O).sum(axis=1)
dQ = np.zeros((N, d))
dK = np.zeros((N, d))
dV = np.zeros((N, d))
cols = list(range(0, N, Bc))
rng.shuffle(cols)
print("column block order:", [c // Bc for c in cols])
for j in cols:
    Kj, Vj = K[j:j + Bc], V[j:j + Bc]
    for i in range(0, N, Br):
        Sij = Q[i:i + Br] @ Kj.T
        Pij = np.exp(Sij - L[i:i + Br, None])
        dV[j:j + Bc] += Pij.T @ dO[i:i + Br]
        dPij = dO[i:i + Br] @ Vj.T
        dSij = Pij * (dPij - D[i:i + Br, None])
        dQ[i:i + Br] += dSij @ Kj          # the shared update: atomic add
        dK[j:j + Bc] += dSij.T @ Q[i:i + Br]

for name, a, b in [("dQ", dQ, dQ_ref), ("dK", dK, dK_ref), ("dV", dV, dV_ref)]:
    print(f"{name}: max difference {np.abs(a - b).max():.1e}")
print("blocks that write dQ_i:", len(cols), "(one per column block)")
