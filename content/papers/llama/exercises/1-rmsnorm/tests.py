import numpy as np
from submission import rmsnorm

X = np.array([1.0, 2.0, 3.0, 4.0])
ONES = np.ones(4)
WANT = np.array([0.3651, 0.7303, 1.0954, 1.4606])


def test_hand_example():
    got = rmsnorm(X, ONES, eps=0.0)
    assert np.allclose(got, WANT, atol=1e-4), f"expected {WANT}, got {got}"


def test_output_has_unit_mean_square():
    rng = np.random.default_rng(0)
    y = rmsnorm(rng.normal(size=(5, 16)) * 7, np.ones(16), eps=0.0)
    ms = np.mean(y**2, axis=-1)
    assert np.allclose(ms, 1.0), f"expected mean square 1 in every row, got {ms}"


def test_scale_invariance():
    a = rmsnorm(X, ONES, eps=1e-9)
    b = rmsnorm(10 * X, ONES, eps=1e-9)
    assert np.allclose(a, b, atol=1e-6), f"expected RMSNorm(10x) == RMSNorm(x), got {a} and {b}"


def test_mean_is_not_subtracted():
    shifted = rmsnorm(X + 10, ONES, eps=0.0)
    want = np.array([0.8765, 0.9562, 1.0359, 1.1155])
    assert np.allclose(shifted, want, atol=1e-4), f"expected {want} (RMSNorm is not shift invariant), got {shifted}"
    assert abs(rmsnorm(X, ONES, eps=0.0).mean()) > 0.5, "the output keeps its mean; a zero mean means you centered the input"


def test_normalizes_the_last_axis():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 8))
    y = rmsnorm(x, np.ones(8), eps=0.0)
    row = rmsnorm(x[1, 2], np.ones(8), eps=0.0)
    assert y.shape == x.shape, f"expected shape {x.shape}, got {y.shape}"
    assert np.allclose(y[1, 2], row), f"each row must be normalized on its own, expected {row}, got {y[1, 2]}"


def test_gain_is_per_feature():
    g = np.array([1.0, 2.0, 0.0, -1.0])
    got = rmsnorm(X, g, eps=0.0)
    want = WANT * g
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_eps_goes_inside_the_square_root():
    got = rmsnorm(np.array([1.0, 1.0]), np.ones(2), eps=1.0)
    want = 1.0 / np.sqrt(2.0)
    assert np.allclose(got, want), f"expected {want:.4f} for each entry, got {got}"
    z = rmsnorm(np.zeros(4), ONES, eps=1e-6)
    assert np.all(np.isfinite(z)) and np.allclose(z, 0), f"expected zeros for a zero input, got {z}"
