"""Token-grid arithmetic for VALL-E (Section 3, Section 4.2, Section 5.1)."""

SAMPLE_RATE = 24_000
HOP = 320  # EnCodec downsampling factor (Section 3)
FRAME_RATE = SAMPLE_RATE // HOP
BITS_PER_CODE = 10  # 1024 entries per codebook


def grid(seconds, quantizers):
    frames = seconds * SAMPLE_RATE // HOP
    return frames, quantizers


print(f"frame rate: {SAMPLE_RATE} / {HOP} = {FRAME_RATE} Hz")
frames, q = grid(10, 8)
print(f"10 s at 8 quantizers: {frames} x {q} = {frames * q} codes")
print(f"bitrate at 8 quantizers: {FRAME_RATE * 8 * BITS_PER_CODE} bit/s")
frames16, _ = grid(10, 16)
print(f"bitrate at 16 quantizers: {FRAME_RATE * 16 * BITS_PER_CODE} bit/s")
print(f"3 s prompt: {3 * FRAME_RATE} frames, {3 * FRAME_RATE * 8} codes")

T = 750
print(f"decoding passes for a 10 s target: AR {T} + NAR 7 = {T + 7}")
print(f"all-AR over 8 codebooks would need {T * 8} passes")
print(f"speedup: {T * 8 / (T + 7):.2f}x")

hours = 60_000
frames_total = hours * 3600 * FRAME_RATE
print(f"60k hours: {frames_total:.3e} frames, {frames_total * 8:.3e} codes")
