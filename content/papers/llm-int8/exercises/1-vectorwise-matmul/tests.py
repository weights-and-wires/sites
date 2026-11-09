import numpy as np
from submission import quantize_rows, vectorwise_matmul

X = np.array([[0.2, -0.4, 0.1], [20.0, 30.0, -10.0]])
W = np.array([[0.5, -1.0], [1.0, 0.3], [-0.2, 0.8]])


def test_quantize_rows_codes():
    A = np.array([[0.3, -1.0], [2.0, 3.0]])
    q, c = quantize_rows(A)
    want = np.array([[38, -127], [85, 127]])
    assert q.dtype == np.int8, f"expected dtype int8, got {q.dtype}"
    assert np.array_equal(q, want), f"expected codes\n{want}\ngot\n{q}"


def test_quantize_rows_constants():
    A = np.array([[0.3, -1.0], [2.0, 3.0]])
    _, c = quantize_rows(A)
    want = np.array([127.0, 127.0 / 3.0])
    assert np.allclose(c, want), f"expected one constant per row {want}, got {c}"


def test_codes_use_the_full_range():
    rng = np.random.default_rng(1)
    q, _ = quantize_rows(rng.normal(size=(5, 40)))
    assert np.all(np.abs(q).max(axis=1) == 127), f"each row's largest code must be +-127, got {np.abs(q).max(axis=1)}"


def test_page_example():
    got = vectorwise_matmul(X, W)
    want = np.array([[-0.318259, -0.240312], [42.071424, -19.070618]])
    assert np.allclose(got, want, atol=1e-4), f"expected {want}, got {got}"


def test_int32_accumulation():
    # 64 terms of 127 * 127 = 1,032,256 overflow int8 and int16.
    Xo, Wo = np.ones((1, 64)), np.ones((64, 1))
    got = vectorwise_matmul(Xo, Wo)
    assert np.allclose(got, [[64.0]], atol=1e-6), f"expected [[64.0]], got {got}; accumulate in int32"


def test_rows_of_x_have_independent_scales():
    got = vectorwise_matmul(X, W)
    exact = X @ W
    err = np.abs(got[0] - exact[0]).max()
    assert err < 0.005, f"row 0 is small and must not share the scale of row 1; expected error < 0.005, got {err:.4f}"


def test_columns_of_w_have_independent_scales():
    Xs = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [3.0, 1.0, -2.0]])
    Ws = np.array([[0.01, 50.0, 0.3], [0.02, -40.0, 0.2], [-0.01, 30.0, -0.1]])
    got = vectorwise_matmul(Xs, Ws)
    exact = Xs @ Ws
    err = np.abs(got[:, 0] - exact[:, 0]).max()
    assert err < 0.002, f"column 0 of W is tiny and needs its own constant; expected error < 0.002, got {err:.4f}"


def test_close_to_exact_on_random_data():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(8, 64)), rng.normal(size=(64, 6))
    got, exact = vectorwise_matmul(A, B), A @ B
    rel = np.linalg.norm(got - exact) / np.linalg.norm(exact)
    assert rel < 0.02, f"expected relative error < 0.02, got {rel:.4f}"
