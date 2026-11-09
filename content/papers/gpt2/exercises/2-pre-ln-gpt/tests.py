import numpy as np
from submission import gpt_block, gpt_forward, residual_scale

D, H, V, CTX = 8, 2, 11, 16


def make_block(rng, scale=0.3, zero=False):
    s = 0.0 if zero else scale
    return {
        "ln1_g": np.ones(D), "ln1_b": np.zeros(D), "ln2_g": np.ones(D), "ln2_b": np.zeros(D),
        "wqkv": rng.standard_normal((D, 3 * D)) * s, "bqkv": rng.standard_normal(3 * D) * s,
        "wo": rng.standard_normal((D, D)) * s, "bo": rng.standard_normal(D) * s,
        "w1": rng.standard_normal((D, 4 * D)) * s, "b1": rng.standard_normal(4 * D) * s,
        "w2": rng.standard_normal((4 * D, D)) * s, "b2": rng.standard_normal(D) * s,
    }


def make_params(seed, n_blocks=2, zero_blocks=False):
    rng = np.random.default_rng(seed)
    return {
        "wte": rng.standard_normal((V, D)), "wpe": rng.standard_normal((CTX, D)) * 0.5,
        "blocks": [make_block(rng, zero=zero_blocks) for _ in range(n_blocks)],
        "lnf_g": np.ones(D), "lnf_b": np.zeros(D),
    }


def test_block_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, D))
    y = gpt_block(x, make_block(rng), H)
    assert y.shape == (5, D), f"expected shape (5, {D}), got {y.shape}"


def test_zero_weights_block_is_identity():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, D)) * 5 + 3
    y = gpt_block(x, make_block(rng, zero=True), H)
    assert np.allclose(y, x), f"a pre-LN block with zero weights must return its input.\nexpected {x[0]}\ngot      {y[0]}"


def test_residual_stream_is_not_normalized():
    rng = np.random.default_rng(2)
    x = np.zeros((3, D))
    x[:, 0] = 100.0
    y = gpt_block(x, make_block(rng, scale=0.05), H)
    assert y[:, 0].min() > 90, f"the 100 on the residual stream should survive; got first column {y[:, 0]}"


def test_block_matches_manual_pre_ln():
    from submission import attention, ffn, layer_norm
    rng = np.random.default_rng(3)
    p = make_block(rng)
    x = rng.standard_normal((6, D))
    mid = x + attention(layer_norm(x, p["ln1_g"], p["ln1_b"]), p, H)
    want = mid + ffn(layer_norm(mid, p["ln2_g"], p["ln2_b"]), p)
    got = gpt_block(x, p, H)
    assert np.allclose(got, want), f"expected\n{want[0]}\ngot\n{got[0]}"


def test_logits_shape():
    out = gpt_forward(np.array([1, 2, 3, 4, 5]), make_params(0), H)
    assert out.shape == (5, V), f"expected shape (5, {V}), got {out.shape}"


def test_final_layer_norm_and_tied_embeddings():
    # wte = identity, no positions, no blocks: logits are the layer-normalized one-hot rows.
    params = {"wte": np.eye(4), "wpe": np.zeros((8, 4)), "blocks": [], "lnf_g": np.ones(4), "lnf_b": np.zeros(4)}
    got = gpt_forward(np.array([0, 2]), params, 1)
    want = np.array([[3 / 4, -1 / 4, -1 / 4, -1 / 4], [-1 / 4, -1 / 4, 3 / 4, -1 / 4]]) / np.sqrt(0.1875 + 1e-5)
    assert np.allclose(got, want, atol=1e-6), f"expected\n{want}\ngot\n{got}"


def test_positions_are_added():
    params = make_params(4, n_blocks=1)
    a = gpt_forward(np.array([3, 3, 3]), params, H)
    assert not np.allclose(a[0], a[1]), "the same token at different positions must get different logits"


def test_causal_logits():
    params = make_params(5)
    a = gpt_forward(np.array([1, 4, 2, 7, 3]), params, H)
    b = gpt_forward(np.array([1, 4, 2, 9, 5]), params, H)
    assert np.allclose(a[:3], b[:3]), "changing tokens 3 and 4 must not change the logits of positions 0 to 2"
    assert not np.allclose(a[3], b[3]), "position 3 reads its own token, so its logits should change"


def test_residual_scale():
    assert abs(residual_scale(12) - 1 / np.sqrt(24)) < 1e-12, f"expected 1/sqrt(24) = {1 / np.sqrt(24):.6f}, got {residual_scale(12)}"
    assert abs(residual_scale(1) - 1 / np.sqrt(2)) < 1e-12, f"expected 1/sqrt(2) = {1 / np.sqrt(2):.6f}, got {residual_scale(1)}"
