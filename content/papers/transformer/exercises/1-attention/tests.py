import numpy as np
from submission import softmax, scaled_dot_product_attention

Q = np.array([[1, 0], [0, 1.0]])
K = np.array([[1, 0], [1, 1.0]])
V = np.array([[1, 2], [3, 4.0]])


def test_stable():
    out = softmax(np.array([1000.0, 0.0]))
    assert np.all(np.isfinite(out)), f"softmax([1000, 0]) = {out}"


def test_weights():
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w[1], [0.3302, 0.6698], atol=1e-4), f"expected [0.3302, 0.6698], got {w[1]}"
    assert np.allclose(out[0], [2, 3]), f"expected [2, 3], got {out[0]}"
