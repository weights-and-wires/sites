import numpy as np
from submission import emergence_index, emergence_size, gains

# Table 2 of the paper, accuracy (%). Sizes are in billions of parameters.
GSM8K = {
    "LaMDA": ([0.42, 2, 8, 68, 137], [2.6, 3.6, 3.2, 5.7, 6.5], [0.4, 1.9, 1.6, 8.2, 14.3]),
    "GPT": ([0.35, 1.3, 6.7, 175], [2.2, 2.4, 4.0, 15.6], [0.5, 0.5, 2.4, 46.9]),
    "PaLM": ([8, 62, 540], [4.9, 9.6, 17.9], [4.1, 29.9, 56.9]),
}
SVAMP_PALM = ([8, 62, 540], [15.1, 48.2, 69.4], [16.8, 46.7, 79.0])


def test_gains_match_the_paper():
    sizes, std, cot = GSM8K["GPT"]
    got = gains(std, cot)
    want = [-1.7, -1.9, -1.6, 31.3]
    assert np.allclose(got, want), f"expected {want}, got {got}"
    sizes, std, cot = GSM8K["PaLM"]
    got = gains(std, cot)
    assert np.allclose(got, [-0.8, 20.3, 39.0]), f"expected [-0.8, 20.3, 39.0], got {got}"


def test_gains_length_mismatch():
    try:
        gains([1, 2], [1])
    except ValueError:
        return
    raise AssertionError("expected ValueError for lists of different lengths")


def test_index_on_a_clean_threshold():
    got = emergence_index([-1.7, -1.9, -1.6, 31.3])
    assert got == 3, f"expected 3, got {got}"
    assert emergence_index([2.0, 3.0, 4.0]) == 0, "all gains positive: expected 0"


def test_a_later_dip_resets_the_index():
    got = emergence_index([1.7, -1.5, 9.6])
    assert got == 2, f"a gain of 1.7 followed by -1.5 is not an emergence at 0; expected 2, got {got}"
    got = emergence_index([-1, 2, -1, 5])
    assert got == 3, f"expected 3, got {got}"


def test_no_emergence():
    assert emergence_index([3.0, 2.0, -0.5]) is None, "the last gain is negative: expected None"
    assert emergence_index([]) is None, "an empty list has no emergence: expected None"


def test_gain_of_exactly_zero_is_not_a_gain():
    got = emergence_index([0.0, 3.0])
    assert got == 1, f"a gain of 0.0 is not above the threshold; expected 1, got {got}"


def test_threshold():
    got = emergence_index([0.3, 5.0, 0.8, 6.0], threshold=1.0)
    assert got == 3, f"expected 3 with threshold 1.0, got {got}"


def test_size_for_each_model_family():
    want = {"LaMDA": 68, "GPT": 175, "PaLM": 62}
    for name, (sizes, std, cot) in GSM8K.items():
        got = emergence_size(sizes, std, cot)
        assert got == want[name], f"{name}: expected {want[name]}B, got {got}"


def test_a_noisy_benchmark_picks_the_last_sign_change():
    sizes, std, cot = SVAMP_PALM
    got = emergence_size(sizes, std, cot)
    assert got == 540, f"PaLM on SVAMP gains +1.7, -1.5, +9.6; expected 540, got {got}"


def test_none_when_the_largest_model_does_not_gain():
    got = emergence_size([1, 2, 3], [10, 20, 30], [12, 22, 29])
    assert got is None, f"expected None, got {got}"
