#!/usr/bin/env python3
"""Last-token pooling with left and right padding (Section 3.2).

Run from the paper folder: python3 scripts/pooling_example.py
"""

import numpy as np

# Two sequences of 4 positions. Sequence 0 has 4 real tokens. Sequence 1 has 2.
# The hidden vector at (b, t) is (10 * b + t, -t), so every position is unique.
B, T, D = 2, 4, 2
hidden = np.zeros((B, T, D))
for b in range(B):
    for t in range(T):
        hidden[b, t] = (10 * b + t, -t)

right = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
left = np.array([[1, 1, 1, 1], [0, 0, 1, 1]])


def last_real_index(mask):
    """Index of the last 1 in each row, which works for any padding side."""
    return mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)


for name, mask in [("right padding", right), ("left padding", left)]:
    idx = last_real_index(mask)
    pooled = hidden[np.arange(B), idx]
    naive = hidden[:, -1]
    print(f"{name}: mask rows {mask.tolist()}")
    print(f"  last real index per row: {idx.tolist()}")
    print(f"  pooled by mask:  {pooled.tolist()}")
    print(f"  naive hidden[:, -1]: {naive.tolist()}")

print("sum(mask) - 1 on left padding:", (left.sum(axis=1) - 1).tolist(), "(wrong for row 1)")
