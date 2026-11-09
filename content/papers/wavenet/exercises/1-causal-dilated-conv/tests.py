import numpy as np
from submission import causal_conv1d


def test_hand_example():
    # Matches the page: x = [1, 2, 3, 4], w = [1, 0.5] (0.5 on the current sample).
    got = causal_conv1d(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 0.5]), 1)
    want = np.array([0.5, 2.0, 3.5, 5.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_dilation_two():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = causal_conv1d(x, np.array([1.0, 1.0]), 2)
    want = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_output_length():
    got = causal_conv1d(np.arange(7.0), np.array([1.0, 2.0, 3.0]), 3)
    assert got.shape == (7,), f"expected shape (7,), got {got.shape}"


def test_dilation_one_matches_numpy_convolve():
    rng = np.random.default_rng(0)
    x, w = rng.normal(size=20), rng.normal(size=3)
    want = np.convolve(x, w[::-1], mode="full")[:20]
    # np.convolve flips the filter, so w[::-1] puts w[-1] on the current sample.
    got = causal_conv1d(x, w, 1)
    assert np.allclose(got, want), f"expected {want[:5]}..., got {got[:5]}..."


def test_filter_order():
    # An impulse at t = 0 shows which tap lands where: tap j lands at (k-1-j)*d.
    x = np.zeros(8)
    x[0] = 1.0
    got = causal_conv1d(x, np.array([1.0, 2.0, 3.0]), 2)
    want = np.array([3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_causality():
    rng = np.random.default_rng(1)
    x = rng.normal(size=30)
    w = rng.normal(size=2)
    base = causal_conv1d(x, w, 4)
    x2 = x.copy()
    x2[20:] += 5.0
    changed = causal_conv1d(x2, w, 4)
    assert np.allclose(base[:20], changed[:20]), (
        f"outputs before t=20 must not change when x[20:] changes; "
        f"max difference {np.abs(base[:20] - changed[:20]).max()}"
    )
    assert not np.allclose(base[20:], changed[20:]), "outputs from t=20 on should change"
