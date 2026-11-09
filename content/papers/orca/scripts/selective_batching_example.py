"""Shapes and costs for selective batching (Section 3, Figure 5).

Run from the paper folder: python3 scripts/selective_batching_example.py
"""

import numpy as np

H = 4
# Figure 5: x1 and x2 are in the increment phase (1 new token each); x3 and x4
# are in the initiation phase with 2 and 3 input tokens.
lens = [1, 1, 2, 3]
ctx = [3, 1, 2, 3]  # tokens each request's attention covers after this iteration
print("new tokens per request:", lens, "-> flat shape", (sum(lens), H))
print("QKV Linear output shape:", (sum(lens), 3 * H))
print("per-request QKV shapes:", [(l, 3 * H) for l in lens])
print("padded [B, L, H] shape:", (len(lens), max(lens), H), "rows =", len(lens) * max(lens), "vs flat rows =", sum(lens))
print("attention score entries per request (new tokens x context):", [l * c for l, c in zip(lens, ctx)])

print()
print("Why skipping the batch for Attention costs little (GPT-3 175B layer, H = 12288, fp16)")
H = 12288
weights = 12 * H * H * 2          # QKV (3H^2) + out (H^2) + MLP (8H^2), 2 bytes each
for context in (128, 512, 2048):
    kv = 2 * context * H * 2      # K and V for one request in one layer
    print(f"context {context:5d}: layer weights {weights / 1e9:.2f} GB (read once per iteration), "
          f"K/V per request {kv / 1e6:.1f} MB (read once per request)")
for bs in (1, 8, 32):
    print(f"batch {bs:2d}: weight bytes per request = {weights / bs / 1e6:.0f} MB")
