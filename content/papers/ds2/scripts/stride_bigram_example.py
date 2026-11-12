"""Compute output lengths for strided convolutions and the CTC frame budget for unigram
and bigram outputs (Section 3.6, Tables 4 and 5).

Run: python3 scripts/stride_bigram_example.py
"""

import math


def to_bigrams(text):
    tokens = []
    for k, word in enumerate(text.split(" ")):
        if k > 0:
            tokens.append(" ")
        for i in range(0, len(word), 2):
            tokens.append(word[i : i + 2])
    return tokens


def min_ctc_frames(tokens):
    return len(tokens) + sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)


def output_length(T, strides):
    for s in strides:
        T = math.ceil(T / s)
    return T


text = "the cat sat"
uni = list(text)
bi = to_bigrams(text)
print(f"text = {text!r}")
print(f"unigrams ({len(uni)}): {uni}")
print(f"bigrams  ({len(bi)}): {bi}")
print(f"minimum CTC frames: unigrams {min_ctc_frames(uni)}, bigrams {min_ctc_frames(bi)}")

print()
print("Repeated letters need a blank between them")
for word in ("hello", "see"):
    print(f"{word!r}: unigram frames {min_ctc_frames(list(word))}, bigram frames {min_ctc_frames(to_bigrams(word))} ({to_bigrams(word)})")

print()
print("Output steps for T = 100 input frames after the paper's Table 4 time strides")
configs = {
    "1-layer 1D (stride 2)": [2],
    "2-layer 1D (strides 1, 2)": [1, 2],
    "3-layer 1D (strides 1, 1, 2)": [1, 1, 2],
    "3-layer 2D (time strides 2, 1, 1)": [2, 1, 1],
}
for name, strides in configs.items():
    print(f"{name:<36} {output_length(100, strides)}")

print("Every Table 4 row shrinks time by 2 overall. The 2-D rows also stride over frequency.")
bins = 161  # 320-point frames, as in spectrogram_example.py
print(f"3-layer 2D frequency strides 2, 2, 2: {bins} bins -> {output_length(bins, [2, 2, 2])} bins")

print()
print("Table 5 strides applied to T = 100 frames, against the frames 'the cat sat' needs")
for s in (2, 3, 4):
    steps = output_length(100, [s])
    print(f"stride {s}: {steps:>3} steps  (unigram need {min_ctc_frames(uni)}, bigram need {min_ctc_frames(bi)})")

print()
print("Toy example: 'the cat sat' squeezed into T = 24 input frames")
for s in (2, 3, 4):
    steps = output_length(24, [s])
    print(f"stride {s}: {steps:>3} steps  unigram fits: {steps >= min_ctc_frames(uni)}  bigram fits: {steps >= min_ctc_frames(bi)}")
