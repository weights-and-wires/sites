#!/usr/bin/env python3
"""Count skipped, masked, and unmasked tiles under a causal mask.

Run: python3 content/papers/fa2/scripts/causal_tiles.py
"""

import math


def counts(N, Br, Bc):
    total = skipped = masked = full = 0
    for i in range(0, N, Br):
        row_min, row_max = i, min(i + Br, N) - 1
        for j in range(0, N, Bc):
            col_min, col_max = j, min(j + Bc, N) - 1
            total += 1
            if col_min > row_max:
                skipped += 1
            elif col_max > row_min:
                masked += 1
            else:
                full += 1
    return total, skipped, masked, full


for N, Br, Bc in [(1024, 128, 128), (1024, 128, 64), (2048, 128, 128), (16384, 128, 128)]:
    total, skipped, masked, full = counts(N, Br, Bc)
    computed = masked + full
    print(f"N={N} Br={Br} Bc={Bc}: total={total} skipped={skipped} masked={masked} full={full} "
          f"computed={computed} bound={total / computed:.3f}")
