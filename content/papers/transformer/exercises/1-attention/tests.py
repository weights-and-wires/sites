import numpy as np
from submission import softmax, scaled_dot_product_attention

Q = np.array([[1, 0], [0, 1.0]])
K = np.array([[1, 0], [1, 1.0]])
V = np.array([[1, 2], [3, 4.0]])


def reference(Q, K, V, mask=None):
    """A slow, loop-based reference that uses the paper's formula directly."""
    d_k = Q.shape[-1]
    n, m = Q.shape[-2], K.shape[-2]
    w = np.zeros(Q.shape[:-2] + (n, m))
    for idx in np.ndindex(*Q.shape[:-2]):
        for i in range(n):
            s = np.array([Q[idx][i] @ K[idx][j] / np.sqrt(d_k) for j in range(m)])
            if mask is not None:
                allowed = np.broadcast_to(mask, w.shape)[idx][i]
                s = np.where(allowed, s, -np.inf)
            s = np.exp(s - s.max())
            w[idx][i] = s / s.sum()
    return w @ V, w


def test_softmax_is_stable():
    out = softmax(np.array([1000.0, 0.0]))
    assert np.all(np.isfinite(out)), f"softmax([1000, 0]) = {out}; subtract the maximum first"
    assert np.allclose(out, [1.0, 0.0]), f"expected [1, 0], got {out}"


def test_softmax_values():
    out = softmax(np.array([2.0, 0.0]))
    assert np.allclose(out, [0.8808, 0.1192], atol=1e-4), f"expected [0.8808, 0.1192], got {out}"


def test_softmax_axis():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 5.0]])
    last = softmax(x, axis=-1)
    assert np.allclose(last.sum(axis=1), 1), f"rows must sum to 1 for axis=-1, got {last.sum(axis=1)}"
    first = softmax(x, axis=0)
    assert np.allclose(first.sum(axis=0), 1), f"columns must sum to 1 for axis=0, got {first.sum(axis=0)}"


def test_tiny_example_by_hand():
    out, w = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(w[0], [0.5, 0.5]), f"row 0 weights: expected [0.5, 0.5], got {w[0]}"
    assert np.allclose(w[1], [0.3302, 0.6698], atol=1e-4), f"row 1 weights: expected [0.3302, 0.6698], got {w[1]}"
    assert np.allclose(out[0], [2, 3]), f"expected [2, 3], got {out[0]}"
    assert np.allclose(out[1], [2.3395, 3.3395], atol=1e-4), f"expected [2.3395, 3.3395], got {out[1]}"


def test_weights_sum_over_keys():
    rng = np.random.default_rng(1)
    q, k, v = rng.normal(size=(3, 4)), rng.normal(size=(5, 4)), rng.normal(size=(5, 2))
    out, w = scaled_dot_product_attention(q, k, v)
    assert w.shape == (3, 5), f"weights shape: expected (3, 5), got {w.shape}"
    assert out.shape == (3, 2), f"output shape: expected (3, 2), got {out.shape}"
    assert np.allclose(w.sum(axis=-1), 1), f"each query's weights must sum to 1 over the keys, got {w.sum(axis=-1)}"


def test_scale_uses_d_k_not_d_v():
    rng = np.random.default_rng(2)
    q, k, v = rng.normal(size=(3, 16)), rng.normal(size=(5, 16)), rng.normal(size=(5, 2))
    out, w = scaled_dot_product_attention(q, k, v)
    want_out, want_w = reference(q, k, v)
    assert np.allclose(w, want_w), f"weights differ from the reference; check the 1/sqrt(d_k) scale with d_k=16\nexpected {want_w[0]}\ngot {w[0]}"
    assert np.allclose(out, want_out), "output differs from the reference"


def test_batched_inputs():
    rng = np.random.default_rng(3)
    q, k, v = rng.normal(size=(2, 3, 4, 8)), rng.normal(size=(2, 3, 6, 8)), rng.normal(size=(2, 3, 6, 5))
    out, w = scaled_dot_product_attention(q, k, v)
    want_out, want_w = reference(q, k, v)
    assert out.shape == (2, 3, 4, 5), f"expected (2, 3, 4, 5), got {out.shape}"
    assert np.allclose(w, want_w), "batched weights differ from the reference; softmax over the key axis (last)"
    assert np.allclose(out, want_out), "batched output differs from the reference"


def test_mask_removes_keys():
    q = np.array([[0.0, 0.0]])
    k = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    v = np.array([[1.0], [2.0], [3.0]])
    mask = np.array([[True, False, True]])
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert np.allclose(w, [[0.5, 0.0, 0.5]], atol=1e-6), f"expected [[0.5, 0, 0.5]], got {w}"
    assert np.allclose(out, [[2.0]]), f"expected [[2.0]], got {out}"


def test_mask_uses_large_negative_not_zero():
    # Every unmasked score is very negative, so a masked score of 0 would win the softmax.
    q = np.array([[1.0, 0.0]])
    k = np.array([[-40.0, 0.0], [-40.0, 0.0], [5.0, 0.0]])
    v = np.array([[1.0], [2.0], [3.0]])
    mask = np.array([[True, True, False]])
    _, w = scaled_dot_product_attention(q, k, v, mask)
    assert w[0, 2] < 1e-6, f"a masked key got weight {w[0, 2]}; replace its score with -1e9, not 0"
    assert np.allclose(w.sum(), 1), f"weights must sum to 1, got {w.sum()}"


def test_mask_broadcasts_over_batch():
    rng = np.random.default_rng(4)
    q, k, v = rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 4)), rng.normal(size=(2, 3, 4))
    mask = np.tril(np.ones((3, 3), dtype=bool))
    out, w = scaled_dot_product_attention(q, k, v, mask)
    want_out, want_w = reference(q, k, v, mask)
    assert np.allclose(w, want_w), "a (n, m) mask must broadcast over the batch axis"
    assert np.allclose(out, want_out), "masked batched output differs from the reference"
