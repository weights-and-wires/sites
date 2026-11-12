"""Pack a sentence pair as [CLS] A [SEP] B [SEP] and sum the token, segment,
and position embeddings (Section 3, Figure 2).

Run: python3 scripts/input_packing.py
"""

import numpy as np

CLS, SEP = 1, 2
a = [10, 11, 12]
b = [20, 21]
ids = [CLS] + a + [SEP] + b + [SEP]
segs = [0] * (len(a) + 2) + [1] * (len(b) + 1)
print(f"ids          : {ids}")
print(f"segment ids  : {segs}")
print(f"positions    : {list(range(len(ids)))}")

# Tiny tables with d = 2 so the sum can be checked by hand.
tok = np.zeros((30, 2))
tok[CLS], tok[SEP], tok[10], tok[11], tok[12], tok[20], tok[21] = (
    [1, 0], [0, 1], [2, 0], [0, 2], [2, 2], [3, 0], [0, 3])
seg = np.array([[0.5, 0.5], [-0.5, -0.5]])
pos = np.arange(16 * 2, dtype=float).reshape(16, 2) / 10
x = tok[ids] + seg[segs] + pos[: len(ids)]
print("input embeddings (token + segment + position), row per position:")
print(np.round(x, 2))
print(f"position 4 ([SEP]) by hand: {tok[SEP]} + {seg[0]} + {pos[4]} = {np.round(tok[SEP] + seg[0] + pos[4], 2)}")
print(f"position 5 (first token of B): {np.round(x[5], 2)}")
