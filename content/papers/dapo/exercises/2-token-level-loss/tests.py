import numpy as np
from submission import objective, token_weights

TERMS = [np.array([1.0, 1.0]), np.full(8, -1.0)]


def test_sample_level_value_from_the_page():
    got = objective(TERMS, "sample")
    assert abs(got - 0.0) < 1e-12, f"expected 0.0, got {got}"


def test_token_level_value_from_the_page():
    got = objective(TERMS, "token")
    assert abs(got - (-0.6)) < 1e-12, f"expected -0.6, got {got}"


def test_levels_agree_when_all_lengths_are_equal():
    terms = [np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 4.0])]
    a, b = objective(terms, "sample"), objective(terms, "token")
    assert abs(a - b) < 1e-12, f"equal lengths should give equal values, got {a} and {b}"


def test_unequal_terms_inside_outputs():
    terms = [np.array([1.0, 3.0]), np.array([2.0, 2.0, 2.0, 8.0])]
    s, t = objective(terms, "sample"), objective(terms, "token")
    assert abs(s - (2.0 + 3.5) / 2) < 1e-12, f"expected sample level 2.75, got {s}"
    assert abs(t - 18.0 / 6) < 1e-12, f"expected token level {18.0 / 6:.4f}, got {t:.4f}"


def test_bad_level_raises():
    try:
        objective(TERMS, "batch")
    except ValueError:
        return
    raise AssertionError("expected ValueError for level 'batch', but nothing was raised")


def test_sample_weights_shrink_with_length():
    w = token_weights([2, 8], "sample")
    assert np.allclose(w[0], 0.25) and len(w[0]) == 2, f"expected two weights of 0.25, got {w[0]}"
    assert np.allclose(w[1], 0.0625) and len(w[1]) == 8, f"expected eight weights of 0.0625, got {w[1]}"


def test_token_weights_are_uniform_and_sum_to_one():
    w = token_weights([2, 8, 5], "token")
    allw = np.concatenate(w)
    assert np.allclose(allw, 1 / 15), f"expected every weight 1/15, got {allw}"
    assert abs(allw.sum() - 1.0) < 1e-12, f"expected weights to sum to 1, got {allw.sum()}"


def test_weights_reproduce_the_objective():
    terms = [np.array([0.5, -1.0, 2.0]), np.array([1.0]), np.array([0.2, 0.4, 0.6, 0.8])]
    for level in ("sample", "token"):
        w = token_weights([len(t) for t in terms], level)
        via = sum(float((wi * ti).sum()) for wi, ti in zip(w, terms))
        direct = objective(terms, level)
        assert abs(via - direct) < 1e-12, f"{level}: weights give {via}, objective gives {direct}"
