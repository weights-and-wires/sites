#!/usr/bin/env python3
"""Bilinear resizing of the position-embedding grid (Section 3.2)."""
import numpy as np

np.set_printoptions(suppress=True)


def coords(g, g_new):
    """Source coordinate of each new index, with the corners aligned."""
    return np.linspace(0, g - 1, g_new)


print("2x2 grid [[0, 1], [2, 3]] resized to 3x3 (corners aligned):")
grid = np.array([[0, 1], [2, 3]], dtype=float)
c = coords(2, 3)
print("  source coordinates:", c.tolist())
out = np.empty((3, 3))
for i, r in enumerate(c):
    for j, s in enumerate(c):
        out[i, j] = (1 - r) * (1 - s) * grid[0, 0] + (1 - r) * s * grid[0, 1] \
            + r * (1 - s) * grid[1, 0] + r * s * grid[1, 1]
print(out)

print("\nA linear ramp f(r, c) = 2r + 3c + 1 on a 3x3 grid, resized to 5x5, stays linear:")
r, cc = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
ramp = 2 * r + 3 * cc + 1.0
c5 = coords(3, 5)
expected = 2 * c5[:, None] + 3 * c5[None, :] + 1
print("  expected value at new (1, 3):", expected[1, 3])
print("  source coordinates:", c5.tolist())

print("\nResolution change for ViT-B/16 (Section 3.2 keeps P fixed)")
for res in (224, 384, 512):
    g = res // 16
    print(f"  {res}x{res}: {g}x{g} grid, {g * g} patches, {g * g + 1} positions")
