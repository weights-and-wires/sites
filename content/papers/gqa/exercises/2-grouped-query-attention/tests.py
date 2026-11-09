import numpy as np
from submission import gqa_attention, scaled_dot_product_attention

n, d, H, d_k = 5, 8, 4, 2
rng = np.random.default_rng(0)
X = rng.normal(size=(n, d))
Wq = rng.normal(size=(d, H * d_k)) / np.sqrt(d)
Wk_all = rng.normal(size=(d, H * d_k)) / np.sqrt(d)
Wv_all = rng.normal(size=(d, H * d_k)) / np.sqrt(d)
Wo = rng.normal(size=(H * d_k, d)) / np.sqrt(H * d_k)
CAUSAL = np.tril(np.ones((n, n), dtype=bool))


def reference(X, Wq, Wk, Wv, Wo, H, G, mask=None):
    """Loop over query heads; head i reads key-value group i // (H // G)."""
    d_k = Wq.shape[1] // H
    outs = []
    for i in range(H):
        g = i // (H // G)
        q = X @ Wq[:, i * d_k:(i + 1) * d_k]
        k = X @ Wk[:, g * d_k:(g + 1) * d_k]
        v = X @ Wv[:, g * d_k:(g + 1) * d_k]
        o, _ = scaled_dot_product_attention(q, k, v, mask)
        outs.append(o)
    return np.concatenate(outs, axis=-1) @ Wo


def kv(G):
    return Wk_all[:, :G * d_k], Wv_all[:, :G * d_k]


def test_shape():
    Wk, Wv = kv(2)
    y = gqa_attention(X, Wq, Wk, Wv, Wo, H, 2)
    assert y.shape == (n, d), f"expected {(n, d)}, got {y.shape}"


def test_two_groups_match_loop():
    Wk, Wv = kv(2)
    got = gqa_attention(X, Wq, Wk, Wv, Wo, H, 2)
    want = reference(X, Wq, Wk, Wv, Wo, H, 2)
    assert np.allclose(got, want), f"GQA-2 differs from the per-head loop; max error {np.abs(got - want).max():.3g}. Heads 0 and 1 share group 0; heads 2 and 3 share group 1."


def test_groups_equal_heads_is_mha():
    got = gqa_attention(X, Wq, Wk_all, Wv_all, Wo, H, H)
    outs = []
    for i in range(H):
        c = slice(i * d_k, (i + 1) * d_k)
        o, _ = scaled_dot_product_attention(X @ Wq[:, c], X @ Wk_all[:, c], X @ Wv_all[:, c])
        outs.append(o)
    want = np.concatenate(outs, axis=-1) @ Wo
    assert np.allclose(got, want), f"G = H must equal multi-head attention; max error {np.abs(got - want).max():.3g}"


def test_one_group_is_mqa():
    Wk, Wv = kv(1)
    got = gqa_attention(X, Wq, Wk, Wv, Wo, H, 1)
    outs = []
    for i in range(H):
        o, _ = scaled_dot_product_attention(X @ Wq[:, i * d_k:(i + 1) * d_k], X @ Wk, X @ Wv)
        outs.append(o)
    want = np.concatenate(outs, axis=-1) @ Wo
    assert np.allclose(got, want), f"G = 1 must share one K and V across all heads; max error {np.abs(got - want).max():.3g}"


def test_neighbors_share_not_interleaved():
    # Changing only group 1's key weights must leave the output of heads 0 and 1 alone.
    Wk, Wv = kv(2)
    Wk2 = Wk.copy()
    Wk2[:, d_k:] += 1.0
    wo_first = np.zeros_like(Wo)
    wo_first[:2 * d_k] = Wo[:2 * d_k]  # keep only heads 0 and 1 in the output
    a = gqa_attention(X, Wq, Wk, Wv, wo_first, H, 2)
    b = gqa_attention(X, Wq, Wk2, Wv, wo_first, H, 2)
    assert np.allclose(a, b), "heads 0 and 1 changed when group 1 changed; query head i must read group i // (H // G), not i % G"


def test_causal_mask():
    Wk, Wv = kv(2)
    y = gqa_attention(X, Wq, Wk, Wv, Wo, H, 2, CAUSAL)
    want = reference(X, Wq, Wk, Wv, Wo, H, 2, CAUSAL)
    assert np.allclose(y, want), f"masked GQA differs from the loop; max error {np.abs(y - want).max():.3g}"
    X2 = X.copy()
    X2[-1] += 5
    y2 = gqa_attention(X2, Wq, Wk, Wv, Wo, H, 2, CAUSAL)
    assert np.allclose(y[:-1], y2[:-1]), "changing the last token changed earlier outputs; the mask leaks the future"


def test_eight_heads_four_groups():
    h8, dk8 = 8, 2
    r = np.random.default_rng(3)
    Wq8 = r.normal(size=(d, h8 * dk8)) / np.sqrt(d)
    Wk8 = r.normal(size=(d, 4 * dk8)) / np.sqrt(d)
    Wv8 = r.normal(size=(d, 4 * dk8)) / np.sqrt(d)
    Wo8 = r.normal(size=(h8 * dk8, d)) / np.sqrt(h8 * dk8)
    got = gqa_attention(X, Wq8, Wk8, Wv8, Wo8, h8, 4)
    want = reference(X, Wq8, Wk8, Wv8, Wo8, h8, 4)
    assert np.allclose(got, want), f"H = 8, G = 4 differs from the loop; max error {np.abs(got - want).max():.3g}"
