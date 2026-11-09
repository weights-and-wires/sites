import numpy as np
from submission import outlier_dims

X = np.array([[0.30, 40.0, -0.50, 0.20],
              [-0.40, 55.0, 0.10, 0.60],
              [0.25, -48.0, 0.35, -0.15]])


def test_page_example():
    got = outlier_dims(X)
    assert list(got) == [1], f"expected [1], got {list(got)}"


def test_returns_columns_not_rows():
    Y = np.zeros((3, 5))
    Y[2, :] = 7.0  # a large token, not a large feature
    Y[:, 3] = 6.5
    got = outlier_dims(Y)
    assert list(got) == [0, 1, 2, 3, 4], f"every column holds a 7.0 in row 2; expected [0, 1, 2, 3, 4], got {list(got)}"
    Z = np.zeros((6, 4))
    Z[1, :] = 9.0
    assert list(outlier_dims(Z)) == [0, 1, 2, 3], f"expected [0, 1, 2, 3], got {list(outlier_dims(Z))}"
    Z2 = np.zeros((4, 6))
    Z2[:, 2] = 7.0
    assert list(outlier_dims(Z2)) == [2], f"expected [2], got {list(outlier_dims(Z2))}"


def test_negative_outliers_count():
    Y = np.zeros((4, 3))
    Y[1, 2] = -8.0
    got = outlier_dims(Y)
    assert list(got) == [2], f"a value of -8.0 has magnitude 8.0; expected [2], got {list(got)}"


def test_one_token_is_enough():
    Y = np.full((1000, 3), 0.1)
    Y[417, 1] = 6.0
    got = outlier_dims(Y)
    assert list(got) == [1], f"one value of magnitude 6.0 in 1000 tokens makes column 1 an outlier; expected [1], got {list(got)}"


def test_threshold_is_inclusive_and_configurable():
    Y = np.array([[5.99, 6.0, 6.01, 3.0]])
    assert list(outlier_dims(Y)) == [1, 2], f"alpha=6.0 and 'at least': expected [1, 2], got {list(outlier_dims(Y))}"
    assert list(outlier_dims(Y, alpha=5.0)) == [0, 1, 2], f"alpha=5.0: expected [0, 1, 2], got {list(outlier_dims(Y, alpha=5.0))}"
    assert list(outlier_dims(Y, alpha=100.0)) == [], f"expected [], got {list(outlier_dims(Y, alpha=100.0))}"


def test_result_is_sorted_integer_array():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(50, 30))
    Y[:, [22, 4, 13]] += 12.0
    got = outlier_dims(Y)
    assert np.issubdtype(np.asarray(got).dtype, np.integer), f"expected an integer array, got dtype {np.asarray(got).dtype}"
    assert list(got) == [4, 13, 22], f"expected [4, 13, 22], got {list(got)}"


def test_empty_result_is_an_integer_array():
    got = np.asarray(outlier_dims(np.ones((3, 4))))
    assert got.shape == (0,) and np.issubdtype(got.dtype, np.integer), f"expected an empty integer array, got shape {got.shape}, dtype {got.dtype}"
