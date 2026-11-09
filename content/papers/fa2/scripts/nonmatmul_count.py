#!/usr/bin/env python3
"""Count non-matmul scalar operations per tile for FlashAttention and FlashAttention-2.

The counting model is this page's, not the paper's. It is the model of
Exercise 2. The paper's own figures are 312 TFLOPs/s of FP16/BF16 matmul and
19.5 TFLOPs/s of FP32 non-matmul on an A100, a ratio of 16 (Section 3.1).

Run: python3 content/papers/fa2/scripts/nonmatmul_count.py
"""

import math


def per_tile(Br, Bc, d, version):
    shared = 4 * Br * Bc
    if version == "fa1":
        return shared + 9 * Br + 3 * Br * d
    return shared + 5 * Br + Br * d


def total(N, d, Br, Bc, version):
    Tr, Tc = math.ceil(N / Br), math.ceil(N / Bc)
    t = Tr * Tc * per_tile(Br, Bc, d, version)
    if version == "fa2":
        t += Tr * (Br * d + 2 * Br)
    return t


print("ratio 312 / 19.5 =", 312 / 19.5)
Br, Bc, d = 128, 64, 64
print(f"\nper tile, Br={Br} Bc={Bc} d={d}")
for v in ("fa1", "fa2"):
    print(f"  {v}: {per_tile(Br, Bc, d, v):,} non-matmul ops")
print("  matmul FLOPs per tile (4*Br*Bc*d):", f"{4 * Br * Bc * d:,}")

print("\nfull forward pass, one head")
for N, d, Br, Bc in [(1024, 64, 128, 64), (4096, 64, 128, 64), (4096, 128, 128, 64)]:
    mm = 4 * N * N * d
    row = [f"N={N} d={d}"]
    shares = []
    for v in ("fa1", "fa2"):
        n = total(N, d, Br, Bc, v)
        share = 16 * n / (16 * n + mm)
        shares.append(share)
        row.append(f"{v}: {n:,} ops, {share:.1%} of time")
    row.append(f"op ratio {total(N, d, Br, Bc, 'fa1') / total(N, d, Br, Bc, 'fa2'):.2f}")
    print("  " + "; ".join(row))
