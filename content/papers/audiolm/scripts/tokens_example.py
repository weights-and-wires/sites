"""Token rates, bitrates, and sequence lengths for AudioLM (Sections III-B, III-C, IV-B).

Run: python3 scripts/tokens_example.py
"""

import math

SR = 16000          # input sample rate, Section III-B
ACOUSTIC_HOP = 320  # SoundStream strides (2, 4, 5, 8), Section IV-B
SEMANTIC_HOP = 640  # w2v-BERT, Section III-B
N = 1024            # codebook size and k-means K
Q = 12              # RVQ layers, Section IV-B
QP = 4              # coarse layers Q', Section IV-B
CHUNK_S = 3         # stage 3 chunk length, Section III-C

acoustic_hz = SR / ACOUSTIC_HOP
semantic_hz = SR / SEMANTIC_HOP
bits = math.log2(N)
print(f"acoustic frame rate = {acoustic_hz:g} Hz, semantic frame rate = {semantic_hz:g} Hz")
print(f"acoustic steps per semantic step = {acoustic_hz / semantic_hz:g}")
print(f"semantic bitrate = {semantic_hz * bits:g} bps")
for q in (QP, Q):
    print(f"acoustic bitrate with {q:2d} quantizers = {acoustic_hz * q * bits:g} bps, {acoustic_hz * q:g} tokens/s")

secs = 10
T = secs * SR
TS = T // SEMANTIC_HOP
TA = T // ACOUSTIC_HOP
print(f"\n10 s clip: T = {T} samples, T_S = {TS}, T_A = {TA}")
print(f"semantic tokens           = {TS}")
print(f"acoustic tokens, Q'={QP}      = {TA * QP}")
print(f"acoustic tokens, Q={Q}      = {TA * Q}")
print(f"waveform samples / semantic tokens = {T / TS:g}")

stage1 = TS
stage2 = TS + TA * QP
chunk_ta = CHUNK_S * SR // ACOUSTIC_HOP
stage3 = chunk_ta * Q  # coarse conditioning plus fine targets, no semantic tokens
chunks = math.ceil(secs / CHUNK_S)
single = TA * Q
interleaved = TS + TA * Q
print(f"\nstage 1 sequence           = {stage1}")
print(f"stage 2 sequence           = {stage2}")
print(f"stage 3 sequence per chunk = {stage3} ({chunks} chunks of {CHUNK_S} s; the last one is {secs - (chunks - 1) * CHUNK_S} s)")
print(f"one stage, acoustic only   = {single}")
print(f"one stage, semantic + all acoustic = {interleaved}")

print("\nSelf-attention cost grows with the square of the length")
for name, n in (("stage 1", stage1), ("stage 2", stage2), ("stage 3 chunk", stage3), ("one stage", interleaved)):
    print(f"{name:14s} n = {n:5d}  n^2 = {n * n:>10,d}")
print(f"ratio one stage / stage 2 = {interleaved**2 / stage2**2:.2f}")

print("\nTokens per semantic token in each stage (the factor of 2 in Figure 2)")
print(f"stage 2: {acoustic_hz / semantic_hz * QP:g}, stage 3: {acoustic_hz / semantic_hz * (Q - QP):g}")
