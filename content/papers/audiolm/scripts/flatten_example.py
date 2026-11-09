"""Flatten a SoundStream token matrix into one stream and invert it (Section III-B).

Symbols here are 0-indexed. The paper counts from 1, so its offset
o_i = ((i - 1) mod Q) * N becomes (i mod Q) * N for a 0-indexed position i.

Run: python3 scripts/flatten_example.py
"""

import numpy as np

N, Q = 1024, 4
Y = np.array([[5, 900, 17, 3], [6, 12, 1000, 4], [5, 7, 2, 1023]])  # T_A = 3 steps, Q = 4 layers

flat = Y.reshape(-1)
offsets = (np.arange(flat.size) % Q) * N
ids = flat + offsets
print("Y shape =", Y.shape)
print("row-major flat =", flat.tolist())
print("offsets        =", offsets.tolist())
print("offset stream  =", ids.tolist())
print("vocabulary size =", Q * N)

back = ids.reshape(-1, Q) - (np.arange(Q) * N)
print("round trip ok  =", bool(np.array_equal(back, Y)))

col = Y.T.reshape(-1)
print("column-major flat (wrong order) =", col.tolist())

print("\nWithout offsets, layer 1 symbol 5 and layer 2 symbol 5 share id 5.")
print("With offsets they become", 5, "and", 5 + N)
