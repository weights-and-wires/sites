#!/usr/bin/env python3
"""Token layout of Decision Transformer (Section 3, Figure 1, Algorithm 1).

K timesteps become 3K tokens in the order R_1, s_1, a_1, ..., R_K, s_K, a_K.
One learned embedding per timestep (not per token) is added to all three
tokens of that timestep.
"""

import numpy as np

K = 3
names = []
timestep = []
for t in range(1, K + 1):
    for m in ("R", "s", "a"):
        names.append(f"{m}_{t}")
        timestep.append(t)

print("token order        :", names)
print("token positions    :", list(range(3 * K)))
print("timestep embedding :", timestep)
print("number of tokens   :", 3 * K)

mask = np.tril(np.ones((3 * K, 3 * K), dtype=bool))
print("\nTokens visible to the state token s_t, which predicts a_t:")
for t in range(1, K + 1):
    idx = 3 * (t - 1) + 1  # position of s_t
    seen = [names[j] for j in range(3 * K) if mask[idx, j]]
    print(f"  s_{t} at position {idx}: sees {len(seen)} tokens: {seen}")

print("\nAttention matrix entries, 3K tokens vs K tokens:")
for k in (1, 20, 30, 50):
    print(f"  K={k:>2}: {(3 * k) ** 2:>6} vs {k ** 2:>5} (ratio {(3 * k) ** 2 // k ** 2})")
