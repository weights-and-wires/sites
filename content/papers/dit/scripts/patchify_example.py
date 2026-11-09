"""Compute the numbers for the patchify example.

Run: python3 scripts/patchify_example.py
"""

import numpy as np


def patchify(z, p):
    """(I, I, C) latent -> (T, p*p*C) tokens, T = (I/p)^2, in raster order."""
    I, _, C = z.shape
    g = I // p
    return z.reshape(g, p, g, p, C).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * C)


def unpatchify(tokens, p, C):
    T = tokens.shape[0]
    g = int(round(T ** 0.5))
    return tokens.reshape(g, g, p, p, C).transpose(0, 2, 1, 3, 4).reshape(g * p, g * p, C)


# A 4x4 single-channel latent holding 0..15.
z = np.arange(16).reshape(4, 4, 1)
print("latent (4x4, C=1):")
print(z[:, :, 0])
tok = patchify(z, 2)
print("p = 2 gives T =", tok.shape[0], "tokens of length", tok.shape[1])
for i, t in enumerate(tok):
    print(f"  token {i}:", t.tolist())
print("round trip exact:", bool(np.array_equal(unpatchify(tok, 2, 1), z)))
print("p = 1 gives T =", patchify(z, 1).shape[0], "; p = 4 gives T =", patchify(z, 4).shape[0])

# Token counts for the paper's 256x256 setup: I = 32, C = 4.
print()
for p in (8, 4, 2):
    T = (32 // p) ** 2
    print(f"I = 32, C = 4, p = {p}: T = {T:4d} tokens, patch vector length p*p*C = {p * p * 4}")
print("512x512 images: I = 64, p = 2: T =", (64 // 2) ** 2)

# Linear embedding parameters do not depend on T, so only the patch width matters.
d = 1152
for p in (8, 4, 2):
    print(f"embedding weights for d = {d}, p = {p}: {p * p * 4 * d}")
