import math


def to_bigrams(text):
    """Split `text` into non-overlapping bigrams inside each word. A space is its own token."""
    raise NotImplementedError


def output_length(T, strides):
    """Number of time steps after a stack of same-padded convolutions with the given time strides."""
    raise NotImplementedError


def min_ctc_frames(tokens):
    """Fewest frames in which CTC can emit `tokens`: one per token, plus one blank between equal neighbors."""
    raise NotImplementedError


def max_stride(T, tokens):
    """Largest integer stride s such that ceil(T / s) frames still hold `tokens` under CTC. Return 0 if none."""
    raise NotImplementedError
