#!/usr/bin/env python3
"""Count HBM reads and writes for standard attention and FlashAttention.

The count follows the line-by-line structure of Algorithm 0 and Algorithm 1
in the FlashAttention paper. The unit is one matrix element. The count
ignores the one-time initialization of O, l, and m (Algorithm 1, line 2).

Run: python3 content/papers/flashattention/scripts/io_count.py
"""

import math


def standard_accesses(N, d):
    """Algorithm 0: S and P go to HBM and come back."""
    line1 = 2 * N * d + N * N        # read Q, K; write S
    line2 = N * N + N * N            # read S; write P
    line3 = N * N + N * d + N * d    # read P, V; write O
    return line1 + line2 + line3


def block_sizes(d, M):
    """Algorithm 1, line 1."""
    Bc = math.ceil(M / (4 * d))
    Br = min(Bc, d)
    return Bc, Br


def flash_accesses(N, d, M, Bc=None):
    """Algorithm 1: K and V once, then Q, O, l, m in and O, l, m out per K/V block."""
    if Bc is None:
        Bc, _ = block_sizes(d, M)
    Tc = math.ceil(N / Bc)
    load_kv = 2 * N * d                         # line 6, summed over j
    load_per_pass = 2 * N * d + 2 * N           # line 8: Q, O, l, m
    write_per_pass = N * d + 2 * N              # line 13 and line 12: O, l, m
    return load_kv + Tc * (load_per_pass + write_per_pass)


N, d, M = 1024, 64, 65536
Bc, Br = block_sizes(d, M)
std = standard_accesses(N, d)
fa = flash_accesses(N, d, M)
print(f"N = {N}, d = {d}, M = {M} elements, Bc = {Bc}, Br = {Br}, Tc = {math.ceil(N / Bc)}, Tr = {math.ceil(N / Br)}")
print(f"standard attention: {std:,} element accesses")
print(f"FlashAttention:     {fa:,} element accesses")
print(f"ratio: {std / fa:.2f}x fewer")
print(f"M / (3 d^2) = {M / (3 * d * d):.2f}")

print()
print("Ratio against N (M = 65536, d = 64):")
for n in (512, 1024, 2048, 4096, 8192):
    print(f"  N = {n:5d}: standard {standard_accesses(n, d):>13,}  flash {flash_accesses(n, d, M):>12,}  ratio {standard_accesses(n, d) / flash_accesses(n, d, M):.2f}")

print()
print("Column block size Bc (N = 1024, d = 64), as in Figure 2 (middle):")
for bc in (64, 128, 256, 512):
    print(f"  Bc = {bc:3d}: Tc = {math.ceil(N / bc):2d}, accesses {flash_accesses(N, d, M, Bc=bc):,}")

print()
print("Operational intensity (fp16, 2 bytes per element, 4 N^2 d FLOPs for the two matmuls):")
flops = 4 * N * N * d
for name, acc in (("standard", std), ("flash", fa)):
    print(f"  {name:9s} {flops / (2 * acc):6.1f} FLOPs per byte")
ridge = 312e12 / 1.5e12
print(f"  ridge point for 312 TFLOP/s and 1.5 TB/s: {ridge:.0f} FLOPs per byte")

print()
print("Extra memory beyond inputs and output (N = 1024, fp16 for S and P, fp32 for l and m):")
print(f"  standard: S and P = {2 * N * N * 2 / 2**20:.1f} MiB per head")
print(f"  flash:    l and m = {2 * N * 4 / 2**10:.1f} KiB per head")

print()
print("Reported in Figure 2 (left), GPT-2 medium, A100 (copied from the paper, not computed):")
paper = {"GFLOPs": (66.6, 75.2), "HBM R/W (GB)": (40.3, 4.4), "Runtime (ms)": (41.7, 7.3)}
for k, (a, b) in paper.items():
    print(f"  {k:14s} standard {a:5.1f}  flash {b:5.1f}  ratio standard/flash {a / b:.2f}")
