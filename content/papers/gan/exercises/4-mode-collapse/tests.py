import numpy as np
from submission import histogram, js_divergence, modes_covered

EDGES = np.linspace(-4, 4, 9)


def mixture(n, rng):
    comp = rng.integers(0, 2, size=n)
    return np.where(comp == 0, -2.0, 2.0) + 0.3 * rng.normal(size=n)


def test_histogram_values():
    got = histogram(np.array([-3.5, -3.5, 0.5, 3.5]), EDGES)
    want = np.array([0.5, 0, 0, 0, 0.25, 0, 0, 0.25])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_histogram_drops_outliers_and_sums_to_one():
    got = histogram(np.array([-100.0, 0.5, 0.5, 100.0]), EDGES)
    assert np.isclose(got.sum(), 1.0), f"the mass must sum to 1 after dropping outliers, got {got.sum()}"
    assert np.isclose(got[4], 1.0), f"expected all mass in bin 4, got {got}"


def test_js_divergence_values():
    p, q = np.array([0.5, 0.5]), np.array([1.0, 0.0])
    want = 0.5 * (0.5 * np.log(0.5 / 0.75) + 0.5 * np.log(0.5 / 0.25)) + 0.5 * (1.0 * np.log(1.0 / 0.75))
    got = js_divergence(p, q)
    assert np.isclose(got, want), f"expected {want:.4f}, got {got}; a zero in q must not produce nan"
    assert np.isclose(got, 0.2158, atol=1e-4), f"expected 0.2158, got {got}"


def test_js_divergence_properties():
    p = np.array([0.1, 0.2, 0.3, 0.4])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.isclose(js_divergence(p, p), 0.0), f"expected 0 for identical inputs, got {js_divergence(p, p)}"
    assert np.isclose(js_divergence(p, q), js_divergence(q, p)), "the JS divergence is symmetric"
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(js_divergence(a, b), np.log(2)), f"expected log 2 for disjoint inputs, got {js_divergence(a, b)}"


def test_collapsed_generator_has_larger_divergence():
    rng = np.random.default_rng(0)
    data, good = mixture(20000, rng), mixture(20000, rng)
    collapsed = 2.0 + 0.3 * rng.normal(size=20000)
    d_good = js_divergence(histogram(data, EDGES), histogram(good, EDGES))
    d_bad = js_divergence(histogram(data, EDGES), histogram(collapsed, EDGES))
    assert d_good < 0.01, f"two samples of the same mixture should have JSD near 0, got {d_good}"
    assert d_bad > 0.15, f"a generator stuck on one mode should have JSD above 0.15, got {d_bad}"


def test_modes_covered_counts_modes():
    rng = np.random.default_rng(1)
    modes = [-2.0, 2.0]
    assert modes_covered(mixture(1000, rng), modes, 1.0, 0.1) == 2, "a balanced mixture covers both modes"
    assert modes_covered(2.0 + 0.3 * rng.normal(size=1000), modes, 1.0, 0.1) == 1, "a collapsed generator covers one mode"


def test_modes_covered_needs_min_frac():
    samples = np.concatenate([np.full(99, 2.0), np.array([-2.0])])
    assert modes_covered(samples, [-2.0, 2.0], 1.0, 0.05) == 1, "one stray sample in 100 is not 5% of the mass"
    assert modes_covered(samples, [-2.0, 2.0], 1.0, 0.01) == 2, "1% of the mass meets min_frac = 0.01"


def test_modes_covered_radius_is_strict():
    samples = np.array([1.0, 1.0, 3.0, 3.0])
    assert modes_covered(samples, [0.0], 1.0, 0.1) == 0, "a sample at distance exactly 1.0 is not within radius 1.0"
    assert modes_covered(samples, [2.0], 1.5, 0.9) == 1, "all four samples are within 1.5 of 2.0"
