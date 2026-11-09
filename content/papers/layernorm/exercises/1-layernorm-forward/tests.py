import numpy as np
from submission import layer_norm

H = 5
ONES, ZEROS = np.ones(H), np.zeros(H)


def test_page_example():
    x = np.array([1.0, 2.0, 3.0, 6.0])
    g = np.array([1.0, 2.0, 1.0, 0.5])
    b = np.array([0.0, 0.5, 0.0, -1.0])
    want = np.array([-1.0690, -0.5690, 0.0, -0.1982])
    got = layer_norm(x, g, b, eps=0.0)
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_zero_mean_unit_variance_per_example():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 5.0, size=(6, H))
    y = layer_norm(x, ONES, ZEROS, eps=0.0)
    assert np.allclose(y.mean(axis=-1), 0, atol=1e-8), f"row means expected 0, got {y.mean(axis=-1)}"
    assert np.allclose(y.var(axis=-1), 1, atol=1e-8), f"row variances expected 1, got {y.var(axis=-1)}"


def test_statistics_are_over_features_not_batch():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(7, H)) * np.arange(1, 8)[:, None]  # rows have very different scales
    y = layer_norm(x, ONES, ZEROS, eps=0.0)
    assert np.allclose(y.std(axis=-1), 1, atol=1e-8), f"each row must have std 1 over its features, got {y.std(axis=-1)}"


def test_uses_population_variance():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    want = (x - 2.5) / np.sqrt(1.25)  # var with divisor H = 4 is 1.25; with H - 1 it would be 1.6667
    got = layer_norm(x, np.ones(4), np.zeros(4), eps=0.0)
    assert np.allclose(got, want, atol=1e-8), f"expected {want} (divide by H), got {got}"


def test_eps_inside_the_square_root():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    want = (x - 2.5) / np.sqrt(1.25 + 0.5)
    got = layer_norm(x, np.ones(4), np.zeros(4), eps=0.5)
    assert np.allclose(got, want, atol=1e-8), f"expected {want}, got {got}"


def test_constant_row_is_finite():
    y = layer_norm(np.full((2, H), 7.0), ONES, np.arange(H, dtype=float), eps=1e-5)
    assert np.all(np.isfinite(y)), f"a constant row must not produce NaN or inf, got {y}"
    assert np.allclose(y, np.tile(np.arange(H, dtype=float), (2, 1))), f"expected the bias, got {y}"


def test_gain_and_bias_are_per_feature():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, H))
    g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([0.5, -0.5, 1.0, -1.0, 2.0])
    base = layer_norm(x, ONES, ZEROS)
    got = layer_norm(x, g, b)
    want = base * g + b
    assert np.allclose(got, want, atol=1e-8), f"expected gain * normalized + bias = {want}, got {got}"


def test_invariant_to_rescaling_and_shifting_an_example():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, H))
    g, b = rng.normal(size=H), rng.normal(size=H)
    a = layer_norm(x, g, b, eps=0.0)
    c = layer_norm(10.0 * x + 3.0, g, b, eps=0.0)
    assert np.allclose(a, c, atol=1e-8), f"Equation 7: expected {a}, got {c}"


def test_each_example_is_independent_of_the_batch():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, H))
    g, b = rng.normal(size=H), rng.normal(size=H)
    full = layer_norm(x, g, b)
    alone = layer_norm(x[2:3], g, b)
    assert np.allclose(full[2], alone[0], atol=1e-10), f"row 2 alone expected {full[2]}, got {alone[0]}"


def test_leading_axes_are_kept():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, H))
    g, b = rng.normal(size=H), rng.normal(size=H)
    y = layer_norm(x, g, b)
    assert y.shape == x.shape, f"expected shape {x.shape}, got {y.shape}"
    want = layer_norm(x[1, 2], g, b)
    assert np.allclose(y[1, 2], want, atol=1e-10), f"position (1, 2) expected {want}, got {y[1, 2]}"
