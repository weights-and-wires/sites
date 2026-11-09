"""Input sizes of the MS-STFT discriminator (Section 3.4, Figure 2).

Assumptions of this script: 1 second of 24 kHz audio, hop = window / 4 (the
label "h=w/4" in Figure 2), and no padding at the ends. The paper does not state
the padding.

Run: python3 scripts/msstft_shapes.py
"""

FS = 24000
N = FS
print("window  hop  frames/s  bins  values (real+imag)  frames per latent step")
for w in (2048, 1024, 512, 256, 128):
    hop = w // 4
    frames = 1 + (N - w) // hop
    bins = w // 2 + 1
    values = 2 * frames * bins
    print(f"{w:6d} {hop:4d} {frames:9d} {bins:5d} {values:19d} {frames / 75:22.2f}")
print(f"\nwaveform samples for comparison: {N}")
print(f"windows in the 48 kHz setup are doubled: {[2 * w for w in (2048, 1024, 512, 256, 128)]}")
