import math

from submission import bitrate_bps, frame_rate, n_streams, sequence_steps


def test_mimi_frame_rate():
    got = frame_rate(24_000, (4, 5, 6, 8, 2))
    assert math.isclose(got, 12.5), f"expected 12.5 Hz, got {got}"


def test_encodec_frame_rate():
    got = frame_rate(24_000, (8, 5, 4, 2))
    assert math.isclose(got, 75.0), f"expected 75.0 Hz for hop 320, got {got}"


def test_strides_multiply():
    got = frame_rate(16_000, (2, 2, 5))
    assert math.isclose(got, 800.0), f"expected 16000 / 20 = 800.0, got {got}"


def test_mimi_bitrate():
    got = bitrate_bps(12.5, 8, 2048)
    assert math.isclose(got, 1100.0), f"expected 1100 bps (12.5 * 8 * 11 bits), got {got}"


def test_bitrate_uses_bits_not_entries():
    got = bitrate_bps(10, 2, 4)
    assert math.isclose(got, 40.0), f"expected 10 * 2 * log2(4) = 40.0, got {got}"


def test_flattened_steps():
    got = sequence_steps(300, 12.5, 8, True)
    assert got == 30_000, f"expected 30000 steps for 5 minutes flattened, got {got}"


def test_temporal_steps():
    got = sequence_steps(300, 12.5, 8, False)
    assert got == 3_750, f"expected 3750 steps for 5 minutes, got {got}"


def test_steps_are_integers():
    got = sequence_steps(1.0, 12.5, 8, False)
    assert got == 12 and isinstance(got, int), f"expected the int 12 (floor of 12.5), got {got!r}"


def test_stream_count():
    assert n_streams(8) == 17, f"expected K = 17 for Q = 8, got {n_streams(8)}"
    assert n_streams(1) == 3, f"expected K = 3 for Q = 1, got {n_streams(1)}"
