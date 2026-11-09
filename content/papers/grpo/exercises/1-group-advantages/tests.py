import numpy as np
from submission import group_advantages


def test_page_example():
    r = np.array([1, 0, 0, 1, 0, 0, 0, 0])
    got = group_advantages(r)
    want = np.array([1.7321, -0.5774, -0.5774, 1.7321, -0.5774, -0.5774, -0.5774, -0.5774])
    assert got.shape == (8,), f"expected shape (8,), got {got.shape}"
    assert np.allclose(got, want, atol=1e-3), f"expected {want}, got {got}"


def test_two_samples_use_population_std():
    got = group_advantages(np.array([1.0, 0.0]))
    assert np.allclose(got, [1.0, -1.0], atol=1e-4), f"std of [1, 0] is 0.5 (ddof=0), so expected [1, -1], got {got}"


def test_zero_mean_unit_std_per_row():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(4, 8)) * np.array([[1], [10], [0.1], [100]]) + np.array([[0], [5], [-3], [50]])
    a = group_advantages(r)
    assert a.shape == (4, 8), f"expected shape (4, 8), got {a.shape}"
    assert np.allclose(a.mean(axis=1), 0, atol=1e-8), f"each row must have mean 0, got {a.mean(axis=1)}"
    assert np.allclose(a.std(axis=1), 1, atol=1e-4), f"each row must have std 1, got {a.std(axis=1)}"


def test_normalizes_over_the_group_axis():
    # 3 questions, group of 5: a wrong axis gives a different shape of statistics.
    r = np.array([[1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 1]], dtype=float)
    a = group_advantages(r)
    row0 = group_advantages(r[0])
    assert np.allclose(a[0], row0, atol=1e-9), f"row 0 must not depend on the other questions: expected {row0}, got {a[0]}"
    assert a[0, 0] > 0 and a[2, 4] > 0, f"the lone correct sample must get a positive advantage, got {a[0, 0]} and {a[2, 4]}"


def test_scale_and_shift_invariance():
    r = np.array([[0.2, 0.9, 0.4, 0.4], [3.0, 1.0, 2.0, 8.0]])
    base = group_advantages(r)
    other = group_advantages(10 * r + 3)
    assert np.allclose(base, other, atol=1e-4), f"advantages must not change when rewards are rescaled; expected {base}, got {other}"


def test_constant_group_gives_zero_not_nan():
    r = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    a = group_advantages(r)
    assert not np.any(np.isnan(a)), f"a group of equal rewards must not produce nan, got {a[0]}"
    assert np.allclose(a[0], 0), f"expected zeros for the constant group, got {a[0]}"
    assert np.allclose(a[1], [-1, 1, -1, 1], atol=1e-4), f"the other row must still be normalized, got {a[1]}"


def test_integer_input_and_no_mutation():
    r = np.array([2, 0, 1, 1])
    r_copy = r.copy()
    a = group_advantages(r)
    assert np.issubdtype(a.dtype, np.floating), f"expected a float array, got {a.dtype}"
    assert np.array_equal(r, r_copy), f"the input must not change: expected {r_copy}, got {r}"
