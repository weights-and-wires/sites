"""Compute the bitrate arithmetic from Section III-A, III-C, and V-D of SoundStream.

Run: python3 scripts/bitrate_example.py
"""

import math

fs = 24000
strides = (2, 4, 5, 8)
M = math.prod(strides)
frame_rate = fs / M
print(f"M = product of strides = {M}")
print(f"frames per second S/T * fs = {frame_rate}")
print(f"frame duration = {1000 * M / fs:.1f} ms")

print("\nPlain VQ at 6 kbps (Section III-C, Limitations of Vector Quantization)")
R = 6000
bits_per_frame = R / frame_rate
print(f"bits per frame r = {bits_per_frame}")
print(f"plain codebook size N = 2^{int(bits_per_frame)} = {2 ** int(bits_per_frame)}")

print("\nResidual VQ at 6 kbps with Nq = 8")
Nq = 8
log2N = bits_per_frame / Nq
N = int(2 ** log2N)
print(f"log2 N = r / Nq = {log2N}, so N = {N}")
print(f"vectors stored = Nq * N = {Nq * N}")
print(f"codes reachable = N^Nq = 2^{int(Nq * log2N)}")

print("\nBitrate per quantizer (bits per frame * frame rate)")
per_q = math.log2(N) * frame_rate
print(f"one 1024-entry quantizer adds {per_q} bps")
for nq in (4, 8, 12, 16, 24):
    print(f"nq = {nq:2d}: {nq * per_q / 1000:g} kbps")

print("\nTable II: three ways to spend 80 bits per frame")
for nq, n in ((8, 1024), (16, 32), (80, 2)):
    bits = nq * math.log2(n)
    print(f"Nq = {nq:2d}, N = {n:4d}: {bits:g} bits per frame, {bits * frame_rate:g} bps, {nq * n} stored vectors")

print("\nTable III: latency and quantizer count at 6 kbps, N = 1024")
for s in ((1, 4, 5, 8), (2, 4, 5, 8), (4, 4, 5, 8)):
    m = math.prod(s)
    rate = fs / m
    nq = R / rate / math.log2(1024)
    print(f"strides {s}: M = {m}, {1000 * m / fs:.1f} ms, {rate:g} frames/s, {R / rate:g} bits/frame, Nq = {nq:g}")

print("\nQuantizer dropout with Nq = 8 (Section III-C)")
Nq = 8
expected_nq = sum(range(1, Nq + 1)) / Nq
print(f"each nq in 1..{Nq} has probability {1 / Nq}")
print(f"expected nq = {expected_nq}")
print(f"expected training bitrate = {expected_nq * per_q / 1000} kbps")
print(f"the first quantizer is used in {100 * 1.0:.0f}% of steps; the last in {100 / Nq:.1f}%")

print("\nRaw audio comparison (assumes 16-bit samples; the paper does not state a sample width)")
raw = fs * 16
print(f"raw rate = {raw} bps = {raw / 1000:g} kbps")
for kbps in (3, 6, 18):
    print(f"{kbps} kbps is {raw / (kbps * 1000):g}x smaller")
