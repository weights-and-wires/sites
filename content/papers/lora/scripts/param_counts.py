"""Compute the parameter and storage counts for the LoRA examples.

Run: python3 scripts/param_counts.py
"""

d_model = 12288  # GPT-3 175B, Section 1 of the paper
layers = 96  # Section 7.1
total = 175e9  # |Phi_0| is about 175 billion (Section 2)

full = d_model * d_model
print(f"one {d_model} x {d_model} matrix: {full:,} weights")
for r in (1, 2, 4, 8):
    per = r * (d_model + d_model)
    print(f"  r={r}: B and A hold {per:,} weights, {full / per:,.0f}x fewer")

print()
for r, mats in [(4, 2), (8, 1), (2, 4), (1, 2)]:
    n = layers * mats * r * 2 * d_model
    print(f"r={r}, {mats} matrices per layer, {layers} layers: {n:,} trainable, {100 * n / total:.4f}% of 175B, {2 * n / 1e6:.1f} MB in FP16")

print()
# Optimizer-state saving in the Adam step: the weights stay, the states shrink.
n = layers * 2 * 4 * 2 * d_model
fp16_model_gb = 2 * total / 1e9
print(f"frozen FP16 weights: {fp16_model_gb:.0f} GB")
print(f"Adam keeps two states per trainable weight; r=4, Wq and Wv: {2 * n:,} state values")
print(f"two states per weight for all of 175B: {2 * total:,.0f} state values")

print()
# Extra multiply-adds per token if the adapter is not merged.
for r in (1, 4, 8):
    extra = r * (d_model + d_model)
    print(f"r={r}: unmerged adapter adds {extra:,} multiply-adds per token per matrix, {100 * extra / full:.4f}% of {full:,}")

print()
# Small example used in the page: d = k = 6, r = 1.
d = k = 6
for r in (1, 2):
    print(f"d=k=6, r={r}: {r * (d + k)} trainable vs {d * k} for a full update")
