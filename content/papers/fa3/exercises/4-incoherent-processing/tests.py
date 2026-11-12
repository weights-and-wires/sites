import numpy as np
from submission import hadamard_transform, incoherent, relative_score_error

H4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2


def sylvester(d):
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(d)


def test_matches_explicit_matrix():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    got = hadamard_transform(x)
    want = x @ H4
    assert np.allclose(got, want), f"expected {want}, got {got}"
    X = np.random.default_rng(0).normal(size=(5, 16))
    assert np.allclose(hadamard_transform(X), X @ sylvester(16)), "expected the 16-point result to match the explicit matrix"


def test_transform_runs_over_the_last_axis():
    X = np.random.default_rng(1).normal(size=(3, 4, 8))
    got = hadamard_transform(X)
    assert got.shape == X.shape, f"expected shape {X.shape}, got {got.shape}"
    assert np.allclose(got, X @ sylvester(8)), "expected each length-8 row to be transformed"


def test_transform_is_orthonormal():
    x = np.random.default_rng(2).normal(size=(4, 32))
    y = hadamard_transform(x)
    assert np.allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1)), "expected the transform to keep each row norm"
    assert np.allclose(hadamard_transform(y), x), "expected applying the transform twice to return the input"


def test_incoherent_order_and_scores():
    rng = np.random.default_rng(3)
    Q, K = rng.normal(size=(6, 16)), rng.normal(size=(9, 16))
    signs = rng.choice([-1.0, 1.0], size=16)
    want = (Q * signs) @ sylvester(16)
    got = incoherent(Q, signs)
    assert np.allclose(got, want), "expected signs to apply before the Hadamard transform"
    assert np.allclose(incoherent(Q, signs) @ incoherent(K, signs).T, Q @ K.T), "expected (QM)(KM)^T = QK^T"


def test_incoherent_spreads_an_outlier():
    x = np.zeros((1, 64))
    x[0, 5] = 100.0
    y = incoherent(x, np.ones(64))
    assert abs(np.abs(y).max() - 100 / 8) < 1e-9, f"expected every entry of size 12.5, got max {np.abs(y).max()}"


def channel_outliers(seed, mag):
    rng = np.random.default_rng(seed)
    Q, K = rng.normal(size=(256, 64)), rng.normal(size=(256, 64))
    Q[:, :2] *= mag
    K[:, :2] *= mag
    return Q, K, rng.choice([-1.0, 1.0], size=64)


def test_rotation_helps_with_outlier_channels():
    for mag in (30, 100):
        Q, K, signs = channel_outliers(0, mag)
        plain = relative_score_error(Q, K, 64)
        rotated = relative_score_error(Q, K, 64, signs)
        assert plain / rotated > 2.5, f"mag={mag}: expected rotation to cut the error 2.5x or more, got plain {plain:.4f}, rotated {rotated:.4f}"


def test_rotation_does_not_help_without_outlier_channels():
    rng = np.random.default_rng(1)
    Q, K = rng.normal(size=(256, 64)), rng.normal(size=(256, 64))
    signs = rng.choice([-1.0, 1.0], size=64)
    plain = relative_score_error(Q, K, 64)
    rotated = relative_score_error(Q, K, 64, signs)
    assert 0.8 < plain / rotated < 1.25, f"expected similar errors on Gaussian data, got plain {plain:.4f}, rotated {rotated:.4f}"
    assert 0.01 < plain < 0.08, f"expected an E4M3 relative error of a few percent, got {plain:.4f}"
