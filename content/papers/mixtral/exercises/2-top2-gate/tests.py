import numpy as np
from submission import top_k_gate

# The logits from the page: one token, 8 experts.
L = np.array([[2.0, 0.5, 1.0, -1.0, 0.0, 1.5, -0.5, 0.2]])


def test_selects_the_two_largest_in_order():
    idx, _ = top_k_gate(L)
    assert idx.shape == (1, 2), f"expected shape (1, 2), got {idx.shape}"
    assert idx[0].tolist() == [0, 5], f"expected experts [0, 5] (logits 2.0 and 1.5), got {idx[0].tolist()}"


def test_weights_match_the_page():
    _, w = top_k_gate(L)
    assert np.allclose(w[0], [0.6225, 0.3775], atol=1e-4), f"expected [0.6225 0.3775], got {w[0]}"


def test_weights_sum_to_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 8))
    _, w = top_k_gate(logits)
    assert np.allclose(w.sum(axis=-1), 1.0), f"every row must sum to 1, got min {w.sum(axis=-1).min()} and max {w.sum(axis=-1).max()}"


def test_equals_renormalized_full_softmax():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(20, 8))
    idx, w = top_k_gate(logits)
    full = np.exp(logits - logits.max(-1, keepdims=True))
    full /= full.sum(-1, keepdims=True)
    picked = np.take_along_axis(full, idx, axis=-1)
    want = picked / picked.sum(-1, keepdims=True)
    assert np.allclose(w, want), f"expected the renormalized full softmax {want[0]}, got {w[0]}"


def test_rows_are_independent():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(6, 8))
    idx, w = top_k_gate(logits)
    idx0, w0 = top_k_gate(logits[:1])
    assert idx[0].tolist() == idx0[0].tolist() and np.allclose(w[0], w0[0]), f"row 0 must not depend on other rows, got {w[0]} vs {w0[0]}"


def test_first_weight_is_the_larger():
    rng = np.random.default_rng(3)
    _, w = top_k_gate(rng.normal(size=(30, 8)))
    assert np.all(w[:, 0] >= w[:, 1]), "the expert with the larger logit must come first and get the larger weight"


def test_k_one_gives_weight_one():
    idx, w = top_k_gate(L, k=1)
    assert idx.shape == (1, 1) and np.allclose(w, 1.0), f"top-1 weight must be 1, got idx {idx} and w {w}"


def test_large_logits_do_not_overflow():
    logits = np.array([[1000.0, 999.0, 0.0, 0.0]])
    idx, w = top_k_gate(logits)
    assert np.all(np.isfinite(w)), f"weights must be finite for logits near 1000, got {w}"
    assert np.allclose(w[0], [0.7311, 0.2689], atol=1e-4), f"expected [0.7311 0.2689], got {w[0]}"
