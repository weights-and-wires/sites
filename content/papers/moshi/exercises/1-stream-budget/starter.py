import math


def frame_rate(sample_rate, strides):
    """Frames per second of an encoder with the given strides."""
    raise NotImplementedError


def bitrate_bps(frame_rate, n_codebooks, codebook_size):
    """Bits per second of an RVQ: every token costs log2(codebook_size) bits."""
    raise NotImplementedError


def sequence_steps(seconds, frame_rate, n_codebooks, flattened):
    """Autoregressive steps for `seconds` of audio."""
    raise NotImplementedError


def n_streams(n_codebooks):
    """K in Equation 6: one text stream plus two audio streams."""
    raise NotImplementedError
