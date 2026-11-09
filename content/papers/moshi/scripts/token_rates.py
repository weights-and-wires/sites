"""Compute the token-rate and sequence-length numbers on the Moshi page.

Run: python3 scripts/token_rates.py
"""

import math

# Mimi encoder (Section 3.3.1): strides (4, 5, 6, 8) and a final stride 2 on 24 kHz audio.
sample_rate = 24_000
strides = (4, 5, 6, 8, 2)
hop = math.prod(strides)
frame_rate = sample_rate / hop
print("hop (samples per frame) =", hop)
print("frame rate (Hz)         =", frame_rate)
print("frame length (ms)       =", 1000 / frame_rate)

# Quantization rate (Section 3.3.1): Q = 8 codebooks of N_A = 2048 entries.
Q, N_A = 8, 2048
bits_per_token = math.log2(N_A)
bitrate = frame_rate * Q * bits_per_token
print("bits per token          =", bits_per_token)
print("bitrate (bps)           =", bitrate)

# WavLM distillation target (Section 3.3.2): 50 Hz, average pooling with stride 4.
print("WavLM rate after pooling =", 50 / 4)

# Sequence lengths (Section 3.4.1): flatten the codebooks or keep them in a depth model.
seconds = 5 * 60
T = int(frame_rate * seconds)
print("tokens per second, flattened, one stream =", frame_rate * Q)
print("steps for 5 min, flattened               =", T * Q)
print("steps for 5 min, Temporal Transformer    =", T)
print("attention pairs ratio (flat / temporal)  =", (T * Q) ** 2 // T**2)

# Joint sequence (Equation 6): K = 2Q + 1 streams.
K = 2 * Q + 1
print("streams K = 2Q + 1                       =", K)
print("tokens per second across all K streams   =", K * frame_rate)
print("steps for 5 min if all K were flattened  =", K * T)
print("Temporal Transformer context 3000 steps  =", 3000 / frame_rate, "s")

# Text stream (Section 3.4.4): about 3 to 4 text tokens per second of speech, 12.5 slots.
for rate in (3, 4):
    print(f"non-PAD fraction at {rate} text tokens/s   =", rate / frame_rate)

# EnCodec comparison, using the encodec page: 24 kHz audio at hop 320.
print("EnCodec frame rate (Hz)  =", 24_000 / 320)
print("Mimi / EnCodec frame-rate ratio =", (24_000 / 320) / frame_rate)
