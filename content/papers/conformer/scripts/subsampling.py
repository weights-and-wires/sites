#!/usr/bin/env python3
"""Frame-rate arithmetic of the convolution subsampling layer (Figure 1)."""

import numpy as np


def out_len(n, k=3, s=2):
    return (n - k) // s + 1


T = 1000  # a 10 s utterance at a 10 ms stride
t1 = out_len(T)
t2 = out_len(t1)
print(f"{T} frames -> {t1} -> {t2} after two stride-2 convolutions (kernel 3, no padding)")
print("frame period: 10 ms x 4 = %d ms, 10 s -> about %d frames" % (10 * 4, 10_000 // 40))
print("attention pairs: %d before, %d after, ratio %.1f" % (T * T, t2 * t2, T * T / (t2 * t2)))

# Frequency axis: 80 filterbank channels through the same two convolutions.
f1 = out_len(80)
f2 = out_len(f1)
print(f"80 mel channels -> {f1} -> {f2}")

# Receptive field of a depthwise kernel of 32 on the subsampled axis.
print("kernel 32 at 40 ms covers %d ms; at 10 ms it would cover %d ms" % (32 * 40, 32 * 10))
