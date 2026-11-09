#!/usr/bin/env python3
"""Arithmetic from Sections 2.3, 2.5 and 6 of the CLIP paper.

    python3 scripts/scale_numbers.py
"""

n = 32768  # minibatch size, Section 2.5
print(f"batch {n}: {n} positive pairs, {n * n - n:,} negative pairs, {n * n:,} similarities")
print(f"float32 similarity matrix: {n * n * 4 / 2**30:.1f} GiB")
print(f"batch 64 with the same recipe: {64 * 64 - 64:,} negatives")

images_seen = 400_000_000 * 32  # 400M pairs, 32 epochs
print(f"\n32 epochs over 400M pairs: {images_seen:,} images seen")
print(f"optimizer steps at batch {n}: {images_seen / n:,.0f}")
seconds_per_year = 365.25 * 24 * 3600
print(f"one image per second: {images_seen / seconds_per_year:.1f} years (Section 6 says 405)")

print("\nsharded similarities with k GPUs: each holds an (n/k) x n block")
for k in (1, 256, 592):
    print(f"  k = {k:>3}: {n // k if n % k == 0 else n / k:.1f} x {n} = {(n / k) * n:,.0f} entries")

print("\nzero-shot ImageNet gains in Section 3.1.4: +1.3 (one prompt), +3.5 (80 prompts), total about +4.8")
print(f"  1.3 + 3.5 = {1.3 + 3.5:.1f}")

print("\nViT patch counts for the CLIP image encoders (patch size P, image size H):")
for name, P, H in [("ViT-B/32", 32, 224), ("ViT-B/16", 16, 224), ("ViT-L/14", 14, 224), ("ViT-L/14@336px", 14, 336)]:
    g = H // P
    print(f"  {name}: {g} x {g} = {g * g} patches, sequence length with a class token {g * g + 1}")

print("\nA 4-pair batch: positives, negatives, loss at scale 0")
import math
print(f"  positives {4}, negatives {4 * 4 - 4}, ln 4 = {math.log(4):.4f}")
