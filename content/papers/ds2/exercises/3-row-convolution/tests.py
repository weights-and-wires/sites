import numpy as np
from submission import row_convolution


def test_tau_zero_scales_each_feature():
    h = np.arange(6, dtype=float).reshape(3, 2)
    W = np.array([[2.0], [10.0]])
    got = row_convolution(h, W)
    want = h * np.array([2.0, 10.0])
    assert np.allclose(got, want), f"expected {want.tolist()}, got {got.tolist()}"


def test_small_hand_computed_example():
    # d = 1, tau = 2, T = 4. h = [1, 2, 3, 4], W = [1, 10, 100].
    h = np.array([[1.0], [2.0], [3.0], [4.0]])
    W = np.array([[1.0, 10.0, 100.0]])
    got = row_convolution(h, W).ravel()
    want = [1 + 20 + 300, 2 + 30 + 400, 3 + 40, 4]
    assert np.allclose(got, want), f"expected {want}, got {got.tolist()}"


def test_looks_at_the_future_not_the_past():
    h = np.zeros((6, 1))
    h[3, 0] = 1.0  # one impulse at t = 3
    W = np.array([[1.0, 2.0, 3.0]])  # tau = 2
    got = row_convolution(h, W).ravel()
    want = [0, 3.0, 2.0, 1.0, 0, 0]  # r_1 = 3 * h_3, r_2 = 2 * h_3, r_3 = 1 * h_3
    assert np.allclose(got, want), f"expected {want}, got {got.tolist()}"


def test_features_do_not_mix():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(7, 3))
    W = rng.normal(size=(3, 4))
    base = row_convolution(h, W)
    h2 = h.copy()
    h2[:, 1] += 5.0
    got = row_convolution(h2, W)
    assert np.allclose(got[:, [0, 2]], base[:, [0, 2]]), "expected features 0 and 2 to ignore a change in feature 1"
    assert not np.allclose(got[:, 1], base[:, 1]), "expected feature 1 to change"


def test_past_input_does_not_affect_the_output():
    rng = np.random.default_rng(2)
    h = rng.normal(size=(8, 2))
    W = rng.normal(size=(2, 3))
    base = row_convolution(h, W)
    h2 = h.copy()
    h2[:4] += 100.0
    got = row_convolution(h2, W)
    # r_t uses h_t .. h_{t+2}. Rows 5 and later only see rows 5 and later.
    assert np.allclose(got[5:], base[5:]), "expected rows 5 and later to ignore changes in rows 0 to 3"


def test_end_of_utterance_is_zero_padded():
    h = np.ones((3, 1))
    W = np.array([[1.0, 1.0, 1.0, 1.0]])  # tau = 3
    got = row_convolution(h, W).ravel()
    assert np.allclose(got, [3.0, 2.0, 1.0]), f"expected [3, 2, 1] from zero padding, got {got.tolist()}"


def test_filter_index_order():
    h = np.array([[1.0], [0.0], [0.0]])
    W = np.array([[1.0, 5.0]])
    got = row_convolution(h, W).ravel()
    # r_0 = W[0] * h_0 + W[1] * h_1 = 1. r_1 = W[0] * h_1 + W[1] * h_2 = 0.
    assert np.allclose(got, [1.0, 0.0, 0.0]), f"expected [1, 0, 0], got {got.tolist()}"
    h2 = np.array([[0.0], [1.0], [0.0]])
    got2 = row_convolution(h2, W).ravel()
    assert np.allclose(got2, [5.0, 1.0, 0.0]), f"expected [5, 1, 0] (W[1] weights the next step), got {got2.tolist()}"


def test_output_shape_and_input_is_not_modified():
    rng = np.random.default_rng(4)
    h = rng.normal(size=(10, 6))
    keep = h.copy()
    got = row_convolution(h, rng.normal(size=(6, 5)))
    assert got.shape == (10, 6), f"expected shape (10, 6), got {got.shape}"
    assert np.array_equal(h, keep), "expected h to stay unchanged"
