import numpy as np
from submission import frame_rate, bitrate, num_quantizers, sample_nq

FS = 24000
STRIDES = (2, 4, 5, 8)


def test_frame_rate():
    got = frame_rate(FS, STRIDES)
    assert np.isclose(got, 75.0), f"expected 75 frames per second (M = 320), got {got}"
    got = frame_rate(FS, (4, 4, 5, 8))
    assert np.isclose(got, 37.5), f"expected 37.5 frames per second (M = 640), got {got}"


def test_bitrate_default():
    got = bitrate(FS, STRIDES, 8, 1024)
    assert np.isclose(got, 6000), f"8 quantizers of 1024 entries at 75 Hz: expected 6000 bps, got {got}"
    got = bitrate(FS, STRIDES, 4, 1024)
    assert np.isclose(got, 3000), f"expected 3000 bps for 4 quantizers, got {got}"


def test_bitrate_table_two():
    for n_q, n in ((8, 1024), (16, 32), (80, 2)):
        got = bitrate(FS, STRIDES, n_q, n)
        assert np.isclose(got, 6000), f"Table II row Nq={n_q}, N={n}: expected 6000 bps, got {got}"


def test_bitrate_table_three():
    for strides, n_q in (((1, 4, 5, 8), 4), ((4, 4, 5, 8), 16)):
        got = bitrate(FS, strides, n_q, 1024)
        assert np.isclose(got, 6000), f"Table III strides {strides}, Nq={n_q}: expected 6000 bps, got {got}"


def test_num_quantizers_exact():
    assert num_quantizers(6000, FS, STRIDES, 1024) == 8, f"6 kbps: expected 8, got {num_quantizers(6000, FS, STRIDES, 1024)}"
    assert num_quantizers(3000, FS, STRIDES, 1024) == 4, f"3 kbps: expected 4, got {num_quantizers(3000, FS, STRIDES, 1024)}"
    assert num_quantizers(18000, FS, STRIDES, 1024) == 24, f"18 kbps: expected 24, got {num_quantizers(18000, FS, STRIDES, 1024)}"


def test_num_quantizers_stays_under_budget():
    got = num_quantizers(7000, FS, STRIDES, 1024)
    assert got == 9, f"7000 bps allows 9 quantizers (6750 bps) but not 10 (7500 bps); got {got}"
    assert num_quantizers(500, FS, STRIDES, 1024) == 0, f"500 bps is below one quantizer (750 bps); expected 0, got {num_quantizers(500, FS, STRIDES, 1024)}"


def test_sample_nq_range_and_uniformity():
    rng = np.random.default_rng(0)
    draws = np.array([sample_nq(8, rng) for _ in range(4000)])
    assert draws.min() == 1, f"smallest draw must be 1, got {draws.min()}"
    assert draws.max() == 8, f"largest draw must be 8 (inclusive), got {draws.max()}"
    counts = np.bincount(draws, minlength=9)[1:]
    assert counts.min() > 380 and counts.max() < 620, f"each value should appear about 500 times; counts for 1..8 were {counts.tolist()}"


def test_sample_nq_degenerate():
    rng = np.random.default_rng(1)
    assert all(sample_nq(1, rng) == 1 for _ in range(20)), "with n_q_max = 1 every draw must be 1"
