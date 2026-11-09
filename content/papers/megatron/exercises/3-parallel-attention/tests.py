import numpy as np
from submission import tp_attention

rng = np.random.default_rng(2)
N, D, H = 6, 16, 8
X = rng.normal(size=(N, D))
Wq, Wk, Wv, Wo = (rng.normal(size=(D, D)) * 0.4 for _ in range(4))
CAUSAL = np.tril(np.ones((N, N), dtype=bool))


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def reference(mask=None):
    """Single-device multi-head attention, one head at a time."""
    d_k = D // H
    outs = []
    for i in range(H):
        s = slice(i * d_k, (i + 1) * d_k)
        scores = (X @ Wq[:, s]) @ (X @ Wk[:, s]).T / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        outs.append(softmax(scores) @ (X @ Wv[:, s]))
    return np.concatenate(outs, axis=1) @ Wo


def test_matches_single_device():
    want = reference()
    for n in (1, 2, 4, 8):
        got = tp_attention(X, Wq, Wk, Wv, Wo, H, n)
        assert got.shape == (N, D), f"n={n}: expected shape {(N, D)}, got {got.shape}"
        assert np.allclose(got, want), f"n={n}: expected max |diff| < 1e-8, got {np.abs(got - want).max()}"


def test_causal_mask_is_forwarded():
    want = reference(CAUSAL)
    got = tp_attention(X, Wq, Wk, Wv, Wo, H, 4, CAUSAL)
    assert np.allclose(got, want), f"expected max |diff| < 1e-8 with a causal mask, got {np.abs(got - want).max()}"


def test_future_tokens_do_not_leak():
    X2 = X.copy()
    X2[-1] += 10.0
    a = tp_attention(X, Wq, Wk, Wv, Wo, H, 2, CAUSAL)
    b = tp_attention(X2, Wq, Wk, Wv, Wo, H, 2, CAUSAL)
    assert np.allclose(a[:-1], b[:-1]), "changing the last token must not change earlier outputs under a causal mask"


def test_rejects_uneven_split():
    try:
        tp_attention(X, Wq, Wk, Wv, Wo, H, 3)
    except ValueError as err:
        assert "divisible" in str(err), f"expected the ValueError message to say 'divisible', got {err!r}"
        return
    raise AssertionError("expected ValueError for 8 heads on 3 GPUs, got a result")


def test_two_heads_per_gpu_keep_the_head_scale():
    # With 4 heads and 2 GPUs, each GPU holds 2 heads of width 4. The scale must still be 1/sqrt(4).
    d, heads = 8, 4
    rng2 = np.random.default_rng(5)
    x = rng2.normal(size=(5, d))
    W = [rng2.normal(size=(d, d)) for _ in range(4)]
    d_k = d // heads
    outs = []
    for i in range(heads):
        s = slice(i * d_k, (i + 1) * d_k)
        sc = (x @ W[0][:, s]) @ (x @ W[1][:, s]).T / np.sqrt(d_k)
        outs.append(softmax(sc) @ (x @ W[2][:, s]))
    want = np.concatenate(outs, axis=1) @ W[3]
    got = tp_attention(x, *W, heads, 2)
    assert np.allclose(got, want), f"expected max |diff| < 1e-8, got {np.abs(got - want).max()}"
