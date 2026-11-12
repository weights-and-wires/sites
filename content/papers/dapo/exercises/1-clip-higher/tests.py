import numpy as np
from submission import clipped_terms, prob_ceiling


def test_positive_advantage_caps_at_the_upper_bound():
    got = clipped_terms(np.array([1.1, 1.25, 1.5]), np.ones(3))
    want = np.array([1.1, 1.25, 1.28])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_ratio_of_1_25_is_clipped_by_the_default_but_not_by_clip_higher():
    default = clipped_terms(np.array([1.25]), np.array([1.0]), eps_low=0.2, eps_high=0.2)
    higher = clipped_terms(np.array([1.25]), np.array([1.0]))
    assert np.allclose(default, [1.2]), f"expected 1.2 with eps_high = 0.2, got {default}"
    assert np.allclose(higher, [1.25]), f"expected 1.25 with eps_high = 0.28, got {higher}"


def test_negative_advantage_uses_the_lower_bound():
    got = clipped_terms(np.array([0.5, 0.75, 1.5]), -np.ones(3))
    want = np.array([-0.8, -0.8, -1.5])
    assert np.allclose(got, want), f"expected {want}, got {got}"


def test_eps_low_and_eps_high_are_separate():
    got = clipped_terms(np.array([0.5, 2.0]), np.array([1.0, -1.0]), eps_low=0.5, eps_high=1.0)
    want = np.array([0.5, -2.0])
    assert np.allclose(got, want), f"expected {want}, got {got}"
    got = clipped_terms(np.array([0.5]), np.array([1.0]), eps_low=0.1, eps_high=1.0)
    assert np.allclose(got, [0.5]), f"a ratio below the range with A > 0 keeps r * A = 0.5, got {got}"


def test_values_inside_the_range_are_the_plain_product():
    r = np.array([0.9, 1.0, 1.1])
    a = np.array([2.0, -3.0, 0.5])
    got = clipped_terms(r, a)
    assert np.allclose(got, r * a), f"expected {r * a}, got {got}"


def test_shape_is_preserved():
    r = np.full((3, 4), 1.5)
    a = np.ones((3, 4))
    got = clipped_terms(r, a)
    assert got.shape == (3, 4), f"expected shape (3, 4), got {got.shape}"


def test_prob_ceiling_matches_the_paper_examples():
    assert np.isclose(prob_ceiling(0.01, eps_high=0.2), 0.012), f"expected 0.012, got {prob_ceiling(0.01, eps_high=0.2)}"
    assert np.isclose(prob_ceiling(0.01), 0.0128), f"expected 0.0128, got {prob_ceiling(0.01)}"
    assert np.isclose(prob_ceiling(0.5), 0.64), f"expected 0.64, got {prob_ceiling(0.5)}"


def test_prob_ceiling_is_capped_at_one():
    got = prob_ceiling(np.array([0.9, 0.99]))
    assert np.allclose(got, [1.0, 1.0]), f"expected [1. 1.], got {got}"
