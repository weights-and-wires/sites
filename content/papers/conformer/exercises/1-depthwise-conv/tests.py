import numpy as np
from submission import depthwise_conv1d


def naive(x, w):
    T, d = x.shape
    k = w.shape[0]
    left = (k - 1) // 2
    y = np.zeros((T, d))
    for t in range(T):
        for c in range(d):
            s = 0.0
            for i in range(k):
                j = t + i - left
                if 0 <= j < T:
                    s += w[i, c] * x[j, c]
            y[t, c] = s
    return y


def test_matches_naive_loop_odd_kernel():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(9, 4))
    w = rng.normal(size=(3, 4))
    got = depthwise_conv1d(x, w)
    want = naive(x, w)
    assert got.shape == (9, 4), f"expected shape (9, 4), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_matches_naive_loop_even_kernel():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 3))
    w = rng.normal(size=(4, 3))
    got = depthwise_conv1d(x, w)
    want = naive(x, w)
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_impulse_is_not_flipped():
    x = np.zeros((7, 1))
    x[3, 0] = 1.0
    w = np.array([[1.0], [2.0], [3.0]])
    got = depthwise_conv1d(x, w)[:, 0]
    want = [0, 0, 3, 2, 1, 0, 0]
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_channels_do_not_mix():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(10, 3))
    w = rng.normal(size=(5, 3))
    base = depthwise_conv1d(x, w)
    x2 = x.copy()
    x2[:, 0] += 1.0
    moved = depthwise_conv1d(x2, w)
    assert np.allclose(base[:, 1:], moved[:, 1:]), "expected channels 1 and 2 to ignore a change in channel 0"
    assert not np.allclose(base[:, 0], moved[:, 0]), "expected channel 0 to change"


def test_kernel_32_is_longer_than_the_input():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(10, 2))
    w = rng.normal(size=(32, 2))
    got = depthwise_conv1d(x, w)
    want = naive(x, w)
    assert got.shape == (10, 2), f"expected shape (10, 2), got {got.shape}"
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_input_is_not_modified():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(6, 2))
    keep = x.copy()
    depthwise_conv1d(x, rng.normal(size=(3, 2)))
    assert np.array_equal(x, keep), "expected the input array to stay unchanged"
