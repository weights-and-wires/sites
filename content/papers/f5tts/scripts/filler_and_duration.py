"""Pad text with filler tokens to the mel length (Equation 5) and estimate
the total duration from the character-count ratio (Section 3.1).

Run: python3 scripts/filler_and_duration.py
"""

SR, HOP = 24000, 256  # Section 4: 24 kHz audio, hop length 256
FPS = SR / HOP
print(f"mel frames per second = {SR} / {HOP} = {FPS}")
print(f"frames in 10 s of speech = {10 * FPS}")

FILLER = "<F>"


def pad_text(chars, n_frames):
    return list(chars) + [FILLER] * (n_frames - len(chars))


z = pad_text("hello", 12)
print()
print("z =", z)
print("M =", 5, "characters, N - M =", z.count(FILLER), "fillers, N =", len(z))

# Duration estimate.
ref_text = "Some call me nature, others call me mother nature."
gen_text = "I am the very model of a modern major general."
ref_frames = 320  # 3.4 seconds of reference mel
print()
print("reference characters:", len(ref_text))
print("generated characters:", len(gen_text))
per_char = ref_frames / len(ref_text)
print(f"frames per character in the reference: {per_char:.4f}")
gen_frames = int(ref_frames * len(gen_text) / len(ref_text))
total = ref_frames + gen_frames
print("generated frames (floor):", gen_frames)
print("total frames N:", total, f"= {total / FPS:.3f} s")
z_all = pad_text(list(ref_text) + list(gen_text), total)
print("characters in z:", len(ref_text) + len(gen_text), "; fillers:", z_all.count(FILLER))

# A clean case that can be checked by hand.
print()
print("40 reference characters over 320 frames, 60 new characters:")
print("total =", 320 + int(320 * 60 / 40))

print()
n4 = int(4 * FPS)
print(f"A 4 s clip has {n4} frames; with 50 characters, fillers = {n4 - 50}")
