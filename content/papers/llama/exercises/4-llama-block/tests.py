import numpy as np
from submission import llama_block

N, D, H, HID = 5, 8, 2, 12


def make_params(seed=0):
    rng = np.random.default_rng(seed)
    w = lambda *s: rng.normal(size=s) * 0.5
    return {
        "g1": np.ones(D), "g2": np.ones(D),
        "Wq": w(D, D), "Wk": w(D, D), "Wv": w(D, D), "Wo": w(D, D),
        "W1": w(D, HID), "V": w(D, HID), "W2": w(HID, D),
    }


def make_x(seed=1):
    return np.random.default_rng(seed).normal(size=(N, D))


def test_shape():
    y = llama_block(make_x(), make_params(), H)
    assert y.shape == (N, D), f"expected shape {(N, D)}, got {y.shape}"


def test_residual_connections():
    p = make_params()
    p["Wo"] = np.zeros((D, D))
    p["W2"] = np.zeros((HID, D))
    x = make_x()
    y = llama_block(x, p, H)
    assert np.allclose(y, x), f"with Wo = 0 and W2 = 0 the block must return x; max difference {np.abs(y - x).max():.3g}"


def test_causal():
    p = make_params()
    x = make_x()
    x2 = x.copy()
    x2[-1] += 5.0
    a, b = llama_block(x, p, H), llama_block(x2, p, H)
    assert np.allclose(a[:-1], b[:-1], atol=1e-8), f"changing the last token changed earlier outputs by {np.abs(a[:-1] - b[:-1]).max():.3g}"
    assert not np.allclose(a[-1], b[-1]), "the last output must depend on the last token"


def test_only_relative_positions_matter():
    p = make_params()
    x = make_x()
    a = llama_block(x, p, H)
    b = llama_block(x, p, H, positions=np.arange(N) + 37)
    assert np.allclose(a, b, atol=1e-8), f"shifting every position by 37 changed the output by {np.abs(a - b).max():.3g}; RoPE on q and k only gives relative positions"


def test_rope_makes_order_matter():
    p = make_params()
    x = make_x()
    swapped = x.copy()
    swapped[[0, 1]] = x[[1, 0]]
    a, b = llama_block(x, p, H)[-1], llama_block(swapped, p, H)[-1]
    assert not np.allclose(a, b, atol=1e-6), "the last token sees the same set of tokens in both orders; with RoPE the outputs differ, but yours match"


def test_pre_norm_attention_branch_ignores_stream_scale():
    p = make_params()
    p["W2"] = np.zeros((HID, D))
    x = make_x()
    a = llama_block(x, p, H) - x
    b = llama_block(10 * x, p, H) - 10 * x
    assert np.allclose(a, b, atol=1e-4), f"the attention update must not depend on the scale of the stream; max difference {np.abs(a - b).max():.3g}"


def test_pre_norm_ffn_branch_ignores_stream_scale():
    p = make_params()
    p["Wo"] = np.zeros((D, D))
    x = make_x()
    a = llama_block(x, p, H) - x
    b = llama_block(10 * x, p, H) - 10 * x
    assert np.allclose(a, b, atol=1e-4), f"the FFN update must not depend on the scale of the stream; max difference {np.abs(a - b).max():.3g}"


def test_seeded_reference_value():
    y = llama_block(make_x(), make_params(), H)
    want = np.array([-3.3402, 0.4293, 2.1161])
    assert np.allclose(y[-1, :3], want, atol=1e-3), f"expected y[-1, :3] = {want}, got {np.round(y[-1, :3], 4)}"
