#!/usr/bin/env python3
"""Thread-block counts for FlashAttention and FlashAttention-2 on an A100.

Simplification: each SM runs one block at a time. A real SM can hold several.
The benchmark setting follows Section 4.1: 16k tokens per batch, hidden
dimension 2048, head dimension 64 (32 heads) or 128 (16 heads). The row block
Br = 128 is a typical choice (Section 3.3), not a value the paper fixes for
every case.

Run: python3 content/papers/fa2/scripts/grid_occupancy.py
"""

import math

SMS = 108
Br = 128


def util(blocks):
    waves = math.ceil(blocks / SMS)
    return waves, blocks / (waves * SMS)


print(f"{'N':>6} {'batch':>5} {'heads':>5} | {'FA1 blocks':>10} {'util':>6} | {'FA2 blocks':>10} {'waves':>5} {'util':>6}")
for heads in (32, 16):
    for N in (512, 2048, 16384):
        batch = 16384 // N
        fa1 = batch * heads
        fa2 = fa1 * math.ceil(N / Br)
        w1, u1 = util(fa1)
        w2, u2 = util(fa2)
        print(f"{N:>6} {batch:>5} {heads:>5} | {fa1:>10} {u1:>6.1%} | {fa2:>10} {w2:>5} {u2:>6.1%}")

print("\nbackward, N=16384 batch=1 heads=16, Bc=128:", 1 * 16 * math.ceil(16384 / 128), "blocks")
