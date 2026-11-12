import math


def to_bigrams(text):
    """Split `text` into non-overlapping bigrams inside each word. A space is its own token."""
    tokens = []
    for k, word in enumerate(text.split(" ")):
        if k > 0:
            tokens.append(" ")
        for i in range(0, len(word), 2):
            tokens.append(word[i : i + 2])
    return tokens


def output_length(T, strides):
    """Number of time steps after a stack of same-padded convolutions with the given time strides."""
    for s in strides:
        T = math.ceil(T / s)
    return T


def min_ctc_frames(tokens):
    """Fewest frames in which CTC can emit `tokens`: one per token, plus one blank between equal neighbors."""
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return len(tokens) + repeats


def max_stride(T, tokens):
    """Largest integer stride s such that ceil(T / s) frames still hold `tokens` under CTC. Return 0 if none."""
    need = min_ctc_frames(tokens)
    best = 0
    for s in range(1, T + 1):
        if output_length(T, [s]) >= need:
            best = s
    return best
