#!/usr/bin/env python3
"""A hand-checkable run of Equation 1 on a 4x4 single-channel image."""
import numpy as np

np.set_printoptions(suppress=True)

img = np.arange(16, dtype=float).reshape(4, 4, 1)  # H=W=4, C=1
P = 2
H, W, C = img.shape
N = H * W // P ** 2
print("image rows:")
print(img[..., 0].astype(int))

patches = img.reshape(H // P, P, W // P, P, C).transpose(0, 2, 1, 3, 4).reshape(N, P * P * C)
print(f"\nN = {N}, flattened patches x_p in R^(N x P^2 C):")
print(patches.astype(int))

wrong = img.reshape(N, P * P * C)
print("\nreshape without the transpose (the bug) gives:")
print(wrong.astype(int))

# D = 2. E picks the sum of the top row of each patch and the sum of the bottom row.
E = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
x_class = np.array([0.5, -0.5])
E_pos = np.array([[0.1, 0.1], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])

proj = patches @ E
print("\nx_p E (patch embeddings):")
print(proj)
z0 = np.vstack([x_class, proj]) + E_pos
print("\nz_0 = [x_class; x_p^1 E; ...; x_p^N E] + E_pos:")
print(z0)
print(f"\nz_0 shape: {z0.shape} = (N+1, D)")

# The same projection as a convolution with kernel P, stride P, and D output channels.
# This equivalence is not stated in the paper; the script checks it numerically.
rng = np.random.default_rng(0)
img2 = rng.normal(size=(8, 8, 3))
P2, D2 = 4, 5
E2 = rng.normal(size=(P2 * P2 * 3, D2))
kernel = E2.reshape(P2, P2, 3, D2)  # (row, col, channel, out) matches the flatten order
conv = np.empty((8 // P2, 8 // P2, D2))
for i in range(8 // P2):
    for j in range(8 // P2):
        window = img2[i * P2:(i + 1) * P2, j * P2:(j + 1) * P2, :]
        conv[i, j] = np.einsum("rcx,rcxd->d", window, kernel)
p2 = img2.reshape(2, P2, 2, P2, 3).transpose(0, 2, 1, 3, 4).reshape(4, -1)
print(f"\nmax |strided conv - patches @ E| = {np.abs(conv.reshape(4, D2) - p2 @ E2).max():.1e}")
