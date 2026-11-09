#!/usr/bin/env python3
"""Compare the split-K (FlashAttention) and split-Q (FlashAttention-2) warp layouts.

One thread block holds a Br x Bc probability tile P and a Bc x d value tile V.
It has W warps. Split-K gives each warp a slice of the Bc columns, so each warp
produces a partial Br x d output, and the partials must be summed through shared
memory. Split-Q gives each warp a slice of the Br rows, so each warp produces
its own output rows. This script runs both layouts, checks that they agree,
and counts the elements that cross warps. The count model (each partial is
written once and read once) is this page's.

Run: python3 content/papers/fa2/scripts/warp_partition.py
"""

import numpy as np

rng = np.random.default_rng(0)
Br, Bc, d, W = 128, 64, 64, 4
P = rng.random((Br, Bc))
V = rng.normal(size=(Bc, d))
want = P @ V

# Split-K: warp w owns columns of P and rows of V.
partials = [P[:, w * Bc // W:(w + 1) * Bc // W] @ V[w * Bc // W:(w + 1) * Bc // W] for w in range(W)]
out_k = sum(partials)
smem_written = sum(p.size for p in partials)
smem_read = sum(p.size for p in partials)

# Split-Q: warp w owns rows of P and the matching rows of the output.
out_q = np.vstack([P[w * Br // W:(w + 1) * Br // W] @ V for w in range(W)])

print(f"Br={Br} Bc={Bc} d={d} warps={W}")
print("split-K equals P @ V:", np.allclose(out_k, want), " split-Q equals P @ V:", np.allclose(out_q, want))
print(f"partial output per warp (split-K): {partials[0].shape}")
print(f"split-K elements written to shared memory: {smem_written:,}")
print(f"split-K elements read back to sum:         {smem_read:,}")
print("split-Q elements exchanged between warps for P @ V: 0")
