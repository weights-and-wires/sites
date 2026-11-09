"""Compute the KV-cache sizes for the 7B-scale example.

The configuration is an assumption for illustration, not a number from the
paper: 32 layers, 32 query heads of width 128 (d_model = 4096), fp16 values,
about 6.74 billion parameters, and an 80 GiB accelerator.

Run: python3 scripts/kv_cache_example.py
"""

L, H, d_k, d_model = 32, 32, 128, 4096
BYTES = 2  # fp16
PARAMS = 6_738_415_616
GIB = 2**30
GPU = 80 * GIB
SEQ = 4096

weights = PARAMS * BYTES
print(f"weights: {weights} bytes = {weights / GIB:.2f} GiB")
print()

configs = [("MHA (G = H = 32)", 32), ("GQA-8", 8), ("MQA (G = 1)", 1)]
for name, G in configs:
    per_token = 2 * L * G * d_k * BYTES  # 2 = one key and one value
    per_seq = per_token * SEQ
    fit = (GPU - weights) // per_seq
    print(f"{name}")
    print(f"  per token  = 2 * {L} * {G} * {d_k} * {BYTES} = {per_token} bytes = {per_token / 1024:g} KiB")
    print(f"  reduction vs MHA: {H // G}x")
    print(f"  per {SEQ}-token sequence = {per_seq} bytes = {per_seq / GIB:g} GiB")
    print(f"  sequences that fit after the weights: {fit}")
    print(f"  batch of 32 at {SEQ} tokens = {32 * per_seq / GIB:g} GiB")
    print()

print("K and V projection weights per layer (W^K and W^V, no biases):")
for name, G in configs:
    n = 2 * d_model * G * d_k
    print(f"  {name}: 2 * {d_model} * {G} * {d_k} = {n}")
