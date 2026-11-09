import numpy as np
from submission import tokens_per_second, bitrate_kbps, n_q_for_kbps, entropy_kbps

STRIDES = (2, 4, 5, 8)


def test_tokens_per_second():
    got = tokens_per_second(24000, STRIDES, 8)
    assert np.isclose(got, 600), f"24 kHz, 8 codebooks: 75 steps * 8 = 600 tokens per second; got {got}"
    got = tokens_per_second(48000, STRIDES, 16)
    assert np.isclose(got, 2400), f"48 kHz, 16 codebooks: 150 steps * 16 = 2400 tokens per second; got {got}"


def test_bitrate_matches_the_paper_targets():
    for n_q, kbps in ((2, 1.5), (4, 3.0), (8, 6.0), (16, 12.0), (32, 24.0)):
        got = bitrate_kbps(24000, STRIDES, n_q)
        assert np.isclose(got, kbps), f"24 kHz with {n_q} codebooks should be {kbps} kbps; got {got}"
    got = bitrate_kbps(48000, STRIDES, 16)
    assert np.isclose(got, 24.0), f"48 kHz with 16 codebooks should be 24 kbps; got {got}"


def test_bitrate_uses_log2_of_the_codebook_size():
    got = bitrate_kbps(24000, STRIDES, 4, n_codes=256)
    assert np.isclose(got, 2.4), f"256 entries are 8 bits: 75 * 4 * 8 = 2400 bps; got {got} kbps"


def test_n_q_for_kbps():
    assert n_q_for_kbps(1.5, 24000, STRIDES) == 2, f"1.5 kbps at 24 kHz is 2 codebooks; got {n_q_for_kbps(1.5, 24000, STRIDES)}"
    assert n_q_for_kbps(3.0, 48000, STRIDES) == 2, f"3 kbps at 48 kHz is 2 codebooks; got {n_q_for_kbps(3.0, 48000, STRIDES)}"
    assert n_q_for_kbps(24.0, 24000, STRIDES) == 32, f"24 kbps at 24 kHz is 32 codebooks; got {n_q_for_kbps(24.0, 24000, STRIDES)}"
    assert isinstance(n_q_for_kbps(6.0, 24000, STRIDES), int), "n_q must be an int"


def test_n_q_for_kbps_rejects_impossible_targets():
    for kbps in (1.0, 2.0, 0.5, 0.0):
        try:
            n_q_for_kbps(kbps, 24000, STRIDES)
        except ValueError:
            continue
        raise AssertionError(f"{kbps} kbps at 24 kHz is not a whole number of 750 bps codebooks; expected ValueError")


def test_uniform_distribution_saves_nothing():
    probs = np.full((8, 1024), 1 / 1024)
    got = entropy_kbps(probs, 24000, STRIDES)
    assert np.isclose(got, 6.0), f"uniform codes cost the full 10 bits each: expected 6.0 kbps, got {got}"


def test_skewed_distribution_by_hand():
    p = np.array([[0.5, 0.25, 0.125, 0.125]])
    got = entropy_kbps(p, 24000, STRIDES)
    want = 1.75 * 75 / 1000
    assert np.isclose(got, want), f"entropy is 1.75 bits; at 75 steps per second expected {want} kbps, got {got}"


def test_zero_probabilities_and_codebook_sum():
    p = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    got = entropy_kbps(p, 24000, STRIDES)
    want = (0.0 + 1.0 + 2.0) * 75 / 1000
    assert np.isfinite(got), f"zero probabilities must not give nan or inf; got {got}"
    assert np.isclose(got, want), f"entropies 0, 1, 2 bits sum to 3 bits per step: expected {want} kbps, got {got}"
