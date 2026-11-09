"""Token and bitrate arithmetic for EnCodec (Sections 3.1, 3.2, 3.3, 4.2, 4.5, 4.6).

Run: python3 scripts/tokens_example.py
"""

import math

BITS = 10  # 1024 entries per codebook (Section 3.2)
STRIDES = (2, 4, 5, 8)  # Section 3.1
M = math.prod(STRIDES)

print(f"M = {M}")
for fs, name in ((24000, "24 kHz"), (48000, "48 kHz")):
    print(f"{name}: {fs / M:g} latent steps per second, {1000 * M / fs:.1f} ms per step")

print("\nBandwidth targets at 24 kHz (75 steps per second, 10 bits per codebook)")
print("n_q  tokens/s  kbps  compression vs 16-bit mono")
for nq in (2, 4, 8, 16, 32):
    tokens = 75 * nq
    kbps = tokens * BITS / 1000
    ratio = 24000 * 16 / (tokens * BITS)
    print(f"{nq:3d}  {tokens:8d}  {kbps:4g}  {ratio:6.0f}x")

print("\n48 kHz stereo (150 steps per second, at most 16 codebooks)")
print("n_q  kbps  compression vs 16-bit stereo")
for nq in (2, 4, 8, 16):
    kbps = 150 * nq * BITS / 1000
    ratio = 48000 * 2 * 16 / (150 * nq * BITS)
    print(f"{nq:3d}  {kbps:4g}  {ratio:6.0f}x")
print(f"MP3 at 64 kbps: {48000 * 2 * 16 / 64000:g}x")

print("\nEntropy coding savings from Table 1 (24 kHz) and Table 4 (48 kHz)")
for name, before, after in (
    ("24 kHz 1.5 kbps", 1.5, 0.9),
    ("24 kHz 3 kbps", 3.0, 1.9),
    ("24 kHz 6 kbps", 6.0, 4.1),
    ("24 kHz 12 kbps", 12.0, 8.9),
    ("48 kHz 6 kbps", 6.0, 4.2),
    ("48 kHz 12 kbps", 12.0, 8.9),
    ("48 kHz 24 kbps", 24.0, 19.4),
):
    print(f"{name}: {before} -> {after} kbps, saves {100 * (1 - after / before):.1f}%")

print("\nUniform codes need log2(1024) = 10 bits; a skewed code needs fewer")
import numpy as np

p = np.array([0.5, 0.25, 0.125, 0.125])
print(f"4-symbol distribution {p.tolist()}: entropy = {-(p * np.log2(p)).sum():.4f} bits (fixed length would be 2)")
# One codebook with 1024 entries where half the probability is on 16 entries.
q = np.concatenate([np.full(16, 0.5 / 16), np.full(1008, 0.5 / 1008)])
h = -(q * np.log2(q)).sum()
print(f"1024-entry codebook, half the mass on 16 entries: entropy = {h:.4f} bits of 10")
print(f"  at 75 steps per second that is {75 * h:.1f} bps for one codebook instead of 750")
