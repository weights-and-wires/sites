"""Tree attention mask and position ids for a small draft tree (Section 3.1,
Figure 6).

Run: python3 scripts/tree_mask_example.py
"""

import numpy as np

# Node i has parent parents[i]; -1 means "child of the last context token".
#   0 --- 2 --- 4
#   |
#   1 --- 3        5 (a second root)
parents = [-1, 0, 0, 1, 2, -1]
names = ["a", "b", "c", "d", "e", "f"]
ctx_len = 3  # context tokens at positions 0, 1, 2

n = len(parents)
depth = np.zeros(n, dtype=int)
for i, p in enumerate(parents):
    depth[i] = 1 if p < 0 else depth[p] + 1
position_ids = ctx_len + depth - 1

tree = np.eye(n, dtype=bool)
for i in range(n):
    j = parents[i]
    while j >= 0:
        tree[i, j] = True
        j = parents[j]
mask = np.concatenate([np.ones((n, ctx_len), dtype=bool), tree], axis=1)

print("depth        :", depth.tolist())
print("position ids :", position_ids.tolist())
print("tree block (rows = query node, columns = key node):")
print("    " + " ".join(names))
for i in range(n):
    print(f"  {names[i]} " + " ".join("1" if v else "." for v in tree[i]))
print("keys visible per node (context + tree):", mask.sum(axis=1).tolist())

# A plain causal chain of the same 6 tokens has a lower-triangular block.
chain = np.tril(np.ones((n, n), dtype=bool))
print("tree block entries set:", int(tree.sum()), "vs causal chain:", int(chain.sum()))
