import numpy as np
from submission import attention

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
I = np.eye(2)


def test_bidirectional_weights_from_the_page():
    out, w = attention(X, I, I, I, causal=False)
    want0 = [0.4011, 0.1978, 0.4011]
    assert np.allclose(w[0], want0, atol=1e-3), f"expected row 0 {want0}, got {np.round(w[0], 4)}"
    assert np.allclose(out[0], [0.8022, 0.5989], atol=1e-3), f"expected output row 0 [0.8022, 0.5989], got {np.round(out[0], 4)}"


def test_causal_weights_from_the_page():
    out, w = attention(X, I, I, I, causal=True)
    want = np.array([[1, 0, 0], [0.3302, 0.6698, 0], [0.2483, 0.2483, 0.5035]])
    assert np.allclose(w, want, atol=1e-3), f"expected {want}, got {np.round(w, 4)}"


def test_rows_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    W = [rng.normal(size=(4, 3)) for _ in range(3)]
    for causal in (False, True):
        _, w = attention(x, *W, causal=causal)
        assert np.allclose(w.sum(axis=1), 1.0), f"causal={causal}: rows must sum to 1, got {w.sum(axis=1)}"


def test_editing_a_later_token_changes_earlier_outputs_only_when_bidirectional():
    x2 = X.copy()
    x2[2] = [-3.0, 2.0]
    a, _ = attention(X, I, I, I, causal=False)
    b, _ = attention(x2, I, I, I, causal=False)
    assert np.abs(a[0] - b[0]).max() > 0.1, "bidirectional: row 0 must move when token 3 changes"
    a, _ = attention(X, I, I, I, causal=True)
    b, _ = attention(x2, I, I, I, causal=True)
    assert np.abs(a[:2] - b[:2]).max() < 1e-12, f"causal: rows 0 and 1 must not move, got change {np.abs(a[:2] - b[:2]).max()}"


def test_future_weights_are_zero_when_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 4))
    W = [rng.normal(size=(4, 4)) for _ in range(3)]
    _, w = attention(x, *W, causal=True)
    top = np.abs(np.triu(w, k=1)).max()
    assert top < 1e-12, f"expected zeros above the diagonal, got max {top}"


def test_scaling_uses_the_projected_width():
    x = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    W = np.zeros((4, 2))
    W[:2, :2] = 2 * np.eye(2)
    _, w = attention(x, W, W, W, causal=False)
    s = np.array([[4.0, 0.0], [0.0, 4.0]]) / np.sqrt(2)
    want = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    assert np.allclose(w, want, atol=1e-9), f"expected scores divided by sqrt(2), the projected width: {want}, got {w}"


def test_large_scores_do_not_overflow():
    x = np.array([[100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    out, w = attention(x, I, I, I, causal=True)
    assert np.all(np.isfinite(out)) and np.all(np.isfinite(w)), f"expected finite values, got {w}"
