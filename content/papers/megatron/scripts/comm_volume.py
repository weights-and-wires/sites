"""Compute communication volume per layer for the paper's 8.3B model.

Run: python3 scripts/comm_volume.py

The ring all-reduce factor 2(t - 1)/t is the standard cost of a bandwidth-optimal
ring all-reduce. The paper doesn't state it.
"""

b, s, h, v, t, layers = 8, 1024, 3072, 51200, 8, 72  # Table 1, Section 5.1
bytes_per = 2  # fp16 activations

tensor = b * s * h
print(f"all-reduced tensor b*s*h = {tensor:,} elements = {tensor * bytes_per / 2**20:.0f} MiB")
per_layer = 4 * tensor
print(f"4 all-reduces per layer (2 forward, 2 backward): {per_layer:,} elements = {per_layer * bytes_per / 2**20:.0f} MiB")
ring = 2 * (t - 1) / t
print(f"ring factor 2(t-1)/t for t = {t}: {ring}")
print(f"per-GPU ring traffic per layer: {per_layer * ring * bytes_per / 2**20:.0f} MiB")
print(f"per-GPU ring traffic, {layers} layers: {layers * per_layer * ring * bytes_per / 2**30:.2f} GiB")

print()
gather = b * s * v
fused = b * s
print(f"logits all-gather b*s*v = {gather:,} elements")
print(f"fused cross-entropy  b*s = {fused:,} elements")
print(f"reduction factor v = {gather // fused:,}")

print()
layer_params = 12 * h * h
emb = v * h
total = layers * layer_params + emb
print(f"params per layer 12 h^2 = {layer_params:,}")
print(f"embedding v*h = {emb:,}")
print(f"total = {total:,} = {total / 1e9:.2f} billion")
print(f"per GPU (layers only, t = {t}) = {layers * layer_params // t:,}")

print()
# Hand-checkable small case: b = 1, s = 2, h = 4.
print("small case b=1, s=2, h=4: per layer", 4 * 1 * 2 * 4, "elements")
