import numpy as np
from submission import semantic_tokens, collapse_repeats

FEATS = np.array([[10.0, 0.1], [30.0, 0.9], [22.0, 0.2]])
MEAN = np.array([20.0, 0.5])
STD = np.array([10.0, 0.4])
CENT = np.array([[-1.0, -1.0], [1.0, 1.0], [0.2, -1.0]])


def test_page_example():
    got = np.asarray(semantic_tokens(FEATS, CENT, MEAN, STD)).tolist()
    assert got == [0, 1, 2], f"expected [0, 1, 2], got {got}"


def test_normalization_matters():
    # Without normalization the large first dimension dominates, and everything maps to centroid 1.
    got = np.asarray(semantic_tokens(FEATS, CENT, MEAN, STD)).tolist()
    assert len(set(got)) == 3, f"the three frames must get three different tokens, got {got}"


def test_nearest_not_farthest():
    feats = np.array([[0.0, 0.0], [5.0, 5.0]])
    cent = np.array([[0.0, 0.0], [5.0, 5.0], [100.0, 100.0]])
    got = np.asarray(semantic_tokens(feats, cent, np.zeros(2), np.ones(2))).tolist()
    assert got == [0, 1], f"expected [0, 1], got {got}"


def test_std_is_used():
    feats = np.array([[4.0, 4.0]])
    cent = np.array([[4.0, 4.0], [1.0, 1.0]])
    got = int(semantic_tokens(feats, cent, np.zeros(2), np.array([4.0, 4.0]))[0])
    assert got == 1, f"4 / 4 = 1 is closest to centroid 1, expected 1, got {got}"


def test_output_type_and_shape():
    got = np.asarray(semantic_tokens(FEATS, CENT, MEAN, STD))
    assert got.shape == (3,) and np.issubdtype(got.dtype, np.integer), f"expected an integer array of shape (3,), got {got.dtype} {got.shape}"


def test_collapse_example():
    kept, runs = collapse_repeats([4, 4, 4, 7, 7, 4, 9])
    assert np.asarray(kept).tolist() == [4, 7, 4, 9], f"expected [4, 7, 4, 9], got {np.asarray(kept).tolist()}"
    assert np.asarray(runs).tolist() == [3, 2, 1, 1], f"expected [3, 2, 1, 1], got {np.asarray(runs).tolist()}"


def test_collapse_keeps_separated_repeats():
    kept, runs = collapse_repeats([1, 2, 1, 2])
    assert np.asarray(kept).tolist() == [1, 2, 1, 2], f"only consecutive repeats go; expected [1, 2, 1, 2], got {np.asarray(kept).tolist()}"
    assert np.asarray(runs).tolist() == [1, 1, 1, 1], f"expected [1, 1, 1, 1], got {np.asarray(runs).tolist()}"


def test_collapse_runs_sum_to_length():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 3, size=40)
    kept, runs = collapse_repeats(t)
    assert int(np.sum(runs)) == 40, f"run lengths must sum to 40, got {np.sum(runs)}"
    assert np.array_equal(np.repeat(kept, runs), t), "repeating each kept token by its run length must rebuild the input"
    assert np.all(np.asarray(kept)[1:] != np.asarray(kept)[:-1]), "no two neighbours may be equal after collapsing"
