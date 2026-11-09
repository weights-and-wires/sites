"""Compute the feature-encoder numbers: total stride, receptive field, frame counts.

Run: python3 scripts/encoder_example.py
Section 4.2: seven blocks, strides (5,2,2,2,2,2,2), kernel widths (10,3,3,3,3,2,2).
"""

import math

KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)
SR = 16000


def out_len(n, kernels=KERNELS, strides=STRIDES):
    for k, s in zip(kernels, strides):
        n = (n - k) // s + 1 if n >= k else 0
    return n


def receptive_field(kernels=KERNELS, strides=STRIDES):
    rf, jump = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * jump
        jump *= s
    return rf, jump


rf, jump = receptive_field()
print("total stride (samples)  =", jump, "=", math.prod(STRIDES))
print("stride in ms            =", jump / SR * 1000)
print("frame rate (Hz)         =", round(SR / jump, 2))
print("receptive field samples =", rf, "=", rf / SR * 1000, "ms")
print("layer-by-layer lengths for 1 s (16000 samples):")
n = SR
for i, (k, s) in enumerate(zip(KERNELS, STRIDES), 1):
    n = (n - k) // s + 1
    print(f"  block {i}: k={k} s={s} -> {n}")
print("frames for 1 s          =", out_len(SR))
print("frames for 250k samples =", out_len(250_000), "(BASE crop, 15.6 s)")
print("naive n // 320 for 1 s  =", SR // 320)
print("naive n // 320 for 250k =", 250_000 // 320)
print("frames for 12.5 s (200000 samples) =", out_len(200_000))
