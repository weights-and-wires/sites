"""Numbers for the "Backup keeps a mean" example.

Run: python3 scripts/backup_example.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

# A two-edge path: edge 0 leaves the root (chosen by player A), edge 1 leaves
# the child (chosen by player B). The leaf is a position where A is to move.
# A leaf value v is from the view of the player to move at the leaf, so the
# edge into the leaf (chosen by B) gets -v and the root edge (chosen by A) gets +v.
# This sign flip is my convention for a two-player zero-sum game.
N = np.zeros(2)
W = np.zeros(2)
leaf_values = [0.4, -0.2, 0.6]
for i, v in enumerate(leaf_values, 1):
    N += 1
    W += np.array([v, -v])
    print(f"after simulation {i}: N = {N}, W = {W}, Q = {W / N}")

# The mean does not care about the order of the simulations.
print("mean of leaf values =", round(float(np.mean(leaf_values)), 4))
