import numpy as np
from submission import llm_int8_matmul, vectorwise_matmul

X = np.array([[0.30, 40.0, -0.50, 0.20],
              [-0.40, 55.0, 0.10, 0.60],
              [0.25, -48.0, 0.35, -0.15]])
W = np.array([[0.5, -0.2], [0.3, 0.4], [-0.6, 0.1], [0.2, 0.7]])


def test_page_example():
    got = llm_int8_matmul(X, W)
    want = np.array([[12.489683, 16.031574], [16.358424, 22.509526], [-14.513938, -19.319211]])
    assert np.allclose(got, want, atol=1e-4), f"expected\n{want}\ngot\n{got}"


def test_beats_plain_vectorwise_with_outliers():
    exact = X @ W
    plain = np.abs(vectorwise_matmul(X, W) - exact).max()
    mixed = np.abs(llm_int8_matmul(X, W) - exact).max()
    assert mixed < plain / 10, f"expected the decomposed error to be under a tenth of {plain:.4f}, got {mixed:.4f}"


def test_no_outliers_equals_vectorwise():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(6, 32)), rng.normal(size=(32, 5))
    assert np.abs(A).max() < 6.0
    got, want = llm_int8_matmul(A, B), vectorwise_matmul(A, B)
    assert np.allclose(got, want, atol=1e-9), f"with no outlier the result must equal vector-wise int8; max diff {np.abs(got - want).max():.2e}"


def test_all_outliers_is_exact_float_matmul():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 5)) + 10.0
    B = rng.normal(size=(5, 3))
    got = llm_int8_matmul(A, B)
    assert np.allclose(got, A @ B, atol=1e-9), f"every column is an outlier, so the result must be the float product; max diff {np.abs(got - A @ B).max():.2e}"


def test_outlier_columns_are_not_double_counted():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(16, 64))
    A[:, 5] = 30.0
    B = rng.normal(size=(64, 8))
    exact = A @ B
    got = llm_int8_matmul(A, B)
    rel = np.linalg.norm(got - exact) / np.linalg.norm(exact)
    assert rel < 0.02, f"expected relative error < 0.02, got {rel:.4f}; check that column 5 is counted once"


def test_outlier_rows_of_w_match_outlier_columns_of_x():
    A = np.array([[0.5, 20.0, 0.5, 0.5], [0.5, 20.0, 0.5, 0.5]])
    B = np.array([[1.0, 1.0], [2.0, -2.0], [1.0, 1.0], [1.0, 1.0]])
    got = llm_int8_matmul(A, B)
    want = A @ B
    assert np.allclose(got, want, atol=0.02), f"expected {want.tolist()}, got {got.tolist()}"


def test_inputs_are_not_modified():
    A, B = X.copy(), W.copy()
    llm_int8_matmul(A, B)
    assert np.array_equal(A, X) and np.array_equal(B, W), "llm_int8_matmul must not change X or W"


def test_shape():
    got = llm_int8_matmul(X, W)
    assert got.shape == (3, 2), f"expected shape (3, 2), got {got.shape}"
