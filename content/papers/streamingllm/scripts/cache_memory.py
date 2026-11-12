"""Worked numbers for "A cache that stops growing".

Run from the paper folder: python3 scripts/cache_memory.py

Geometry is Llama-2-7B's (32 layers, 32 KV heads of width 128, fp16). These
are the page's assumptions, not numbers from the paper. The cache of 4 + 2044
slots is the Llama-2 setting of Table 6.
"""

layers, kv_heads, head_dim, bytes_per = 32, 32, 128, 2
per_token = 2 * kv_heads * head_dim * layers * bytes_per
print("KV bytes per token:", per_token, f"= {per_token // 1024} KiB")

slots = 4 + 2044
print("streaming cache slots:", slots, f"= {slots * per_token / 2**30:.2f} GiB, for every n above {slots}")
for n in (2048, 4096, 20_000, 400_000, 4_000_000):
    full = n * per_token / 2**30
    stream = min(n, slots) * per_token / 2**30
    print(f"n={n:>9,}: full cache {full:10.2f} GiB | streaming {stream:5.2f} GiB | ratio {full / stream:8.1f}")

# Sliding window with recomputation re-runs the window for every new token.
for L in (1024, 2048, 4096):
    pairs_recompute = L * (L + 1) // 2
    pairs_stream = L
    print(f"L={L}: attention pairs per new token: recompute {pairs_recompute:,} | cached {pairs_stream:,} | ratio {pairs_recompute / pairs_stream:.1f}")
