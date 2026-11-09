#!/usr/bin/env python3
"""Sequence lengths, attention sizes, and the patch projection E (Section 3.1)."""

H = W = 224
C = 3


def n_patches(h, w, p):
    assert h % p == 0 and w % p == 0
    return (h // p) * (w // p)


print("Patch counts N = HW/P^2 (Section 3.1)")
for h, p in [(224, 16), (224, 14), (224, 32), (384, 16), (512, 16)]:
    n = n_patches(h, h, p)
    print(f"  {h}x{h}, P={p}: N={n}, sequence length N+1={n + 1}, grid {h // p}x{h // p}")

print("\nFlattened patch width P^2*C and the size of E in R^{(P^2*C) x D}")
for name, p, d in [("ViT-B/16", 16, 768), ("ViT-L/16", 16, 1024), ("ViT-H/14", 14, 1280)]:
    width = p * p * C
    print(f"  {name}: P^2*C={width}, D={d}, E has {width * d:,} weights")

print("\nAttention score entries per head, per image")
n_pix = H * W
n_vit = n_patches(H, W, 16) + 1
print(f"  pixels as tokens: {n_pix:,} tokens -> {n_pix ** 2:,} scores")
print(f"  ViT-B/16 patches: {n_vit} tokens -> {n_vit ** 2:,} scores")
print(f"  ratio: {n_pix ** 2 / n_vit ** 2:,.0f}x")

print("\nSequence length grows with the inverse square of P (Section 4.1)")
for p in (32, 16, 8):
    print(f"  224x224, P={p}: N={n_patches(224, 224, p)}")
