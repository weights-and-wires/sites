"""Worked numbers for "The problem" and "Near-zero fragmentation".

Run from the paper folder: python3 scripts/fragmentation_example.py

Part 1 uses the paper's own numbers: the KV cache size of OPT-13B (Section 3)
and the two requests of Figure 3. Part 2 is this page's own simulation with
lengths drawn from a distribution that this script defines, so only its shape
carries over to the paper.
"""
import math

import numpy as np

# Part 1: OPT-13B (Section 3, "Large KV cache").
hidden, layers, fp16_bytes = 5120, 40, 2
per_token = 2 * hidden * layers * fp16_bytes
print("OPT-13B KV cache per token (bytes):", per_token, f"= {per_token / 1024:.0f} KiB")
print("one 2048-token request:", f"{per_token * 2048 / 2**30:.2f} GiB", f"= {per_token * 2048 / 1e9:.2f} GB")
print("Table 1, 13B: 12 GiB of KV memory holds", f"{12 * 2**30 // per_token} token slots")
print("12 GiB / one 2048-token request =", f"{12 * 2**30 / (per_token * 2048):.1f} full-length requests")

# Part 1b: Figure 3 on the contiguous layout, and the same requests with blocks.
B = 16
for name, max_len, prompt, generated in [("A", 2048, 7, 1), ("B", 512, 3, 1)]:
    used = prompt + generated
    print(f"\nFigure 3 request {name}: max {max_len}, {prompt} prompt + {generated} generated")
    print(f"  contiguous: reserves {max_len}, uses {used}, wastes {max_len - used}")
    blocks = math.ceil(used / B)
    print(f"  paged B={B}: {blocks} block(s) = {blocks * B} slots, wastes {blocks * B - used}")

# Part 2: how many requests fit in 15.7K slots (Table 1, 13B column).
capacity = 15728          # = 12 GiB / 800 KiB, the paper's "15.7K" slots
max_len = 2048
rng = np.random.default_rng(0)
# Own distribution: total length (prompt + output) lognormal, median ~300, clipped to [8, 2048].
lengths = np.clip(rng.lognormal(mean=math.log(300), sigma=0.9, size=20000), 8, max_len).astype(int)
print(f"\nSimulated lengths: mean {lengths.mean():.1f}, median {np.median(lengths):.0f}, max {lengths.max()}")

def admit_contiguous(lengths, per_request):
    n, used = 0, 0
    for _ in lengths:
        if used + per_request > capacity:
            break
        used += per_request
        n += 1
    return n

def admit_paged(lengths, B):
    n, used = 0, 0
    for L in lengths:
        need = math.ceil(L / B) * B
        if used + need > capacity:
            break
        used += need
        n += 1
    return n

n_max = admit_contiguous(lengths, max_len)
print(f"reserve the maximum ({max_len}): {n_max} requests at once")
pow2 = lambda L: 1 << (int(L) - 1).bit_length()
n, used = 0, 0
for L in lengths:
    need = pow2(L)
    if used + need > capacity:
        break
    used += need
    n += 1
print(f"reserve next power of two: {n} requests at once")
n, used = 0, 0
for L in lengths:
    if used + L > capacity:
        break
    used += L
    n += 1
print(f"oracle, exact lengths: {n} requests at once")
for B in (1, 4, 16, 64, 256):
    print(f"paged B={B:>3}: {admit_paged(lengths, B)} requests at once")

print("\nInternal waste per request, mean over all simulated requests")
for B in (1, 4, 16, 64, 256):
    waste = (np.ceil(lengths / B) * B - lengths).mean()
    print(f"  B={B:>3}: {waste:6.2f} slots (bound: at most B-1 = {B - 1}; (B-1)/2 = {(B - 1) / 2:.1f})")
print(f"  contiguous max: {(max_len - lengths).mean():.1f} slots")
