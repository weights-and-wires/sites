import numpy as np
from submission import decoder_inputs, perplexity, sequence_log_prob

LOGITS = np.array([
    [2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 3.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 2.0],
])
TARGET = [0, 1, 2, 3, 4]


def test_decoder_inputs_shift_right():
    got = decoder_inputs([5, 6, 7, 4], 4)
    assert list(got) == [4, 5, 6, 7], f"expected [4, 5, 6, 7], got {list(got)}"


def test_decoder_inputs_same_length_and_figure_1():
    got = decoder_inputs([0, 1, 2, 3, 4], 4)
    assert len(got) == 5 and list(got) == [4, 0, 1, 2, 3], f"expected [4, 0, 1, 2, 3], got {list(got)}"


def test_log_prob_value():
    got = sequence_log_prob(LOGITS, TARGET)
    assert np.isclose(got, -2.8216, atol=1e-3), f"expected -2.8216, got {got}"


def test_log_prob_is_a_log_probability():
    got = sequence_log_prob(LOGITS, TARGET)
    assert got < 0, f"a sequence log probability must be negative, got {got}"
    assert np.isclose(np.exp(got), 0.059512, atol=1e-5), f"expected exp(log p) = 0.059512, got {np.exp(got)}"


def test_softmax_axis_on_non_square_logits():
    logits = np.array([[0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    got = sequence_log_prob(logits, [1, 0])
    want = 2 * (5.0 - np.log(np.exp(5.0) + 2.0))
    assert np.isclose(got, want), f"expected {want}, got {got}"


def test_stable_for_large_logits():
    logits = np.array([[1000.0, 0.0], [0.0, 1000.0]])
    got = sequence_log_prob(logits, [0, 1])
    assert np.isfinite(got) and abs(got) < 1e-6, f"expected about 0.0, got {got}"


def test_sums_not_averages():
    got = sequence_log_prob(np.zeros((4, 2)), [0, 1, 0, 1])
    want = 4 * np.log(0.5)
    assert np.isclose(got, want), f"expected {want} (sum over 4 steps), got {got}"


def test_perplexity():
    got = perplexity(-2.8216, 5)
    assert np.isclose(got, 1.7582, atol=1e-3), f"expected 1.7582, got {got}"
    assert np.isclose(perplexity(4 * np.log(0.5), 4), 2.0), f"a uniform 2-way model has perplexity 2, got {perplexity(4 * np.log(0.5), 4)}"
