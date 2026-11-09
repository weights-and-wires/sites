import numpy as np
from submission import multi_head_attention, scaled_dot_product_attention

n, d, h = 5, 8, 2
rng = np.random.default_rng(0)
X = rng.normal(size=(n, d))
Wq, Wk, Wv, Wo = [rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]
CAUSAL = np.tril(np.ones((n, n), dtype=bool))


def per_head_reference(X, Wq, Wk, Wv, Wo, h, mask=None):
    """Loop over heads. Head i uses column block i of each projection (W_i^Q, W_i^K, W_i^V)."""
    d_model = X.shape[1]
    d_k = d_model // h
    outs = []
    for i in range(h):
        cols = slice(i * d_k, (i + 1) * d_k)
        o, _ = scaled_dot_product_attention(X @ Wq[:, cols], X @ Wk[:, cols], X @ Wv[:, cols], mask)
        outs.append(o)
    return np.concatenate(outs, axis=-1) @ Wo


def test_shape():
    y = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    assert y.shape == (n, d), f"expected {(n, d)}, got {y.shape}"


def test_matches_per_head_loop():
    got = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    want = per_head_reference(X, Wq, Wk, Wv, Wo, h)
    assert np.allclose(got, want), f"differs from the per-head loop; max error {np.abs(got - want).max():.3g}. Check reshape then transpose to (h, n, d_k)."


def test_matches_loop_with_four_heads_and_mask():
    got = multi_head_attention(X, Wq, Wk, Wv, Wo, 4, CAUSAL)
    want = per_head_reference(X, Wq, Wk, Wv, Wo, 4, CAUSAL)
    assert np.allclose(got, want), f"h=4 with a causal mask differs from the loop; max error {np.abs(got - want).max():.3g}"


def test_one_head_is_plain_attention():
    got = multi_head_attention(X, Wq, Wk, Wv, Wo, 1)
    want = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)[0] @ Wo
    assert np.allclose(got, want), "with h=1 the result must equal single-head attention followed by Wo"


def test_causality():
    y = multi_head_attention(X, Wq, Wk, Wv, Wo, h, CAUSAL)
    X2 = X.copy()
    X2[-1] += 5
    y2 = multi_head_attention(X2, Wq, Wk, Wv, Wo, h, CAUSAL)
    assert np.allclose(y[:-1], y2[:-1]), "changing the last token changed earlier outputs; the mask leaks the future"
    assert not np.allclose(y[-1], y2[-1]), "the last output should change when the last token changes"


def test_permutation_equivariance_without_positions():
    p = np.random.default_rng(1).permutation(n)
    y = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    yp = multi_head_attention(X[p], Wq, Wk, Wv, Wo, h)
    assert np.allclose(yp, y[p]), "shuffling the input rows must shuffle the output rows the same way"


def test_mask_changes_output():
    free = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    masked = multi_head_attention(X, Wq, Wk, Wv, Wo, h, CAUSAL)
    assert not np.allclose(free, masked), "the causal mask had no effect; is the mask passed to the attention helper?"
