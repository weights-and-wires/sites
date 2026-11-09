import numpy as np
from submission import causal_mask, scaled_dot_product_attention

Q = np.array([[1, 0], [0, 1.0]])
K = np.array([[1, 0], [1, 1.0]])
V = np.array([[1, 2], [3, 4.0]])


def test_mask_type_and_shape():
    m = causal_mask(4)
    assert isinstance(m, np.ndarray) and m.dtype == bool, f"expected a boolean array, got {getattr(m, 'dtype', type(m))}"
    assert m.shape == (4, 4), f"expected shape (4, 4), got {m.shape}"


def test_mask_values():
    want = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    got = causal_mask(3)
    assert np.array_equal(got, want), f"expected\n{want.astype(int)}\ngot\n{np.asarray(got).astype(int)}"


def test_diagonal_is_visible():
    m = causal_mask(5)
    assert np.all(np.diag(m)), f"each position must attend to itself; diagonal is {np.diag(m)}"
    assert m[0].sum() == 1, f"row 0 must see exactly one key, sees {m[0].sum()}"


def test_one_by_one():
    assert np.array_equal(causal_mask(1), np.array([[True]])), f"expected [[True]], got {causal_mask(1)}"


def test_attention_with_mask():
    _, w = scaled_dot_product_attention(Q, K, V, causal_mask(2))
    assert np.allclose(w[0], [1, 0]), f"row 0 expected [1, 0], got {w[0]}"
    assert np.allclose(w[1], [0.3302, 0.6698], atol=1e-4), f"row 1 expected [0.3302, 0.6698], got {w[1]}"


def test_future_gets_no_weight():
    rng = np.random.default_rng(0)
    n = 6
    q, k, v = rng.normal(size=(n, 4)), rng.normal(size=(n, 4)), rng.normal(size=(n, 4))
    _, w = scaled_dot_product_attention(q, k, v, causal_mask(n))
    future = np.triu(w, k=1)
    assert np.allclose(future, 0, atol=1e-8), f"weights above the diagonal must be 0, got max {future.max()}"
    assert np.allclose(w.sum(-1), 1), f"rows must still sum to 1, got {w.sum(-1)}"
