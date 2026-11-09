import math


def frame_rate(sample_rate, strides):
    """Frames per second of an encoder with the given strides."""
    return sample_rate / math.prod(strides)


def bitrate_bps(frame_rate, n_codebooks, codebook_size):
    """Bits per second of an RVQ: every token costs log2(codebook_size) bits."""
    return frame_rate * n_codebooks * math.log2(codebook_size)


def sequence_steps(seconds, frame_rate, n_codebooks, flattened):
    """Autoregressive steps for `seconds` of audio."""
    frames = int(seconds * frame_rate)
    return frames * n_codebooks if flattened else frames


def n_streams(n_codebooks):
    """K in Equation 6: one text stream plus two audio streams."""
    return 2 * n_codebooks + 1
