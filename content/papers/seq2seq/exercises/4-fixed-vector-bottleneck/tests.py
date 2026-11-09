import numpy as np
from submission import V, encode, experiment, fit_readout, token_accuracy


def test_encode_by_hand():
    A = 2.0 * np.eye(2)
    E = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 5.0]])
    got = encode([0, 1, 2], E, A)
    # h1 = [1, 0]; h2 = 2*h1 + [0, 1] = [2, 1]; h3 = 2*h2 + [3, 5] = [7, 7]
    assert np.allclose(got, [7.0, 7.0]), f"expected [7, 7], got {got}"


def test_encode_order_matters():
    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    E = np.eye(2)
    a, b = encode([0, 1], E, A), encode([1, 0], E, A)
    assert not np.allclose(a, b), f"different orders must give different states, got {a} and {b}"
    assert np.allclose(a, [0.0, 2.0]) and np.allclose(b, [0.0, 0.0]), f"expected [0, 2] and [0, 0], got {a} and {b}"


def test_fit_readout_recovers_exact_map():
    rng = np.random.default_rng(1)
    Hs = rng.normal(size=(50, 5))
    W_true = rng.normal(size=(5, 3))
    W = fit_readout(Hs, Hs @ W_true, 1e-9)
    assert W.shape == (5, 3), f"expected shape (5, 3), got {W.shape}"
    assert np.allclose(W, W_true, atol=1e-4), f"expected W_true, max error {np.abs(W - W_true).max()}"


def test_ridge_shrinks_the_solution():
    rng = np.random.default_rng(2)
    Hs, Y = rng.normal(size=(40, 4)), rng.normal(size=(40, 2))
    small, big = fit_readout(Hs, Y, 1e-6), fit_readout(Hs, Y, 100.0)
    assert np.linalg.norm(big) < np.linalg.norm(small), f"larger ridge must shrink W: {np.linalg.norm(big)} vs {np.linalg.norm(small)}"


def test_token_accuracy_counts_positions():
    Hs = np.eye(2)
    W = np.zeros((2, 2 * V))
    for pos, tok in enumerate([1, 3]):
        W[0, pos * V + tok] = 1.0
    for pos, tok in enumerate([0, 2]):
        W[1, pos * V + tok] = 1.0
    tokens = np.array([[1, 3], [0, 0]])
    got = token_accuracy(W, Hs, tokens)
    assert np.isclose(got, 0.75), f"expected 0.75 (3 of 4 positions), got {got}"


def test_short_sources_fit_in_the_state():
    got = experiment(4)
    assert got > 0.99, f"n=4 with H=24: expected accuracy above 0.99, got {got:.3f}"


def test_accuracy_falls_with_length():
    short, mid, long_ = experiment(6), experiment(12), experiment(24)
    assert short > mid > long_, f"expected accuracy to fall with length, got {short:.3f}, {mid:.3f}, {long_:.3f}"
    assert long_ < 0.8, f"n=24 with H=24 expected below 0.8, got {long_:.3f}"
