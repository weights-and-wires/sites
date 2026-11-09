import numpy as np
from submission import selective_batch_layer

H = 4
rng = np.random.default_rng(7)
Wqkv = rng.normal(size=(H, 3 * H))
Wo = rng.normal(size=(H, H))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def reference_single(x):
    """One request on its own, written independently of the submission."""
    q, k, v = x @ Wqkv[:, :H], x @ Wqkv[:, H:2 * H], x @ Wqkv[:, 2 * H:]
    L = len(x)
    s = q @ k.T / np.sqrt(H)
    for i in range(L):
        s[i, i + 1:] = -1e9
    return softmax(s) @ v @ Wo


def run(lens, seed=1):
    x = np.random.default_rng(seed).normal(size=(sum(lens), H))
    return x, selective_batch_layer(x, lens, Wqkv, Wo)


def test_shape():
    _, out = run([1, 1, 2, 3])
    assert out.shape == (7, H), f"expected shape (7, {H}), got {out.shape}"


def test_matches_each_request_alone():
    lens = [1, 1, 2, 3]
    x, out = run(lens)
    start = 0
    for r, n in enumerate(lens):
        want = reference_single(x[start:start + n])
        got = out[start:start + n]
        assert np.allclose(got, want, atol=1e-6), f"request {r} differs from running it alone:\nexpected {want}\ngot {got}"
        start += n


def test_requests_do_not_see_each_other():
    lens = [2, 3]
    x, out = run(lens)
    x2 = x.copy()
    x2[:2] += 5.0  # change request 0 only
    out2 = selective_batch_layer(x2, lens, Wqkv, Wo)
    assert np.allclose(out[2:], out2[2:], atol=1e-8), f"request 1's output changed when only request 0 changed; max diff {np.abs(out[2:] - out2[2:]).max()}"


def test_attention_is_causal_within_a_request():
    lens = [3, 2]
    x, out = run(lens)
    x2 = x.copy()
    x2[2] += 5.0  # last token of request 0
    out2 = selective_batch_layer(x2, lens, Wqkv, Wo)
    assert np.allclose(out[:2], out2[:2], atol=1e-8), f"earlier tokens must not see later ones; max diff {np.abs(out[:2] - out2[:2]).max()}"


def test_equal_lengths_and_single_request():
    for lens in ([3, 3], [5]):
        x, out = run(lens, seed=3)
        want = np.concatenate([reference_single(x[a:b]) for a, b in zip(np.cumsum([0] + lens[:-1]), np.cumsum(lens))])
        assert np.allclose(out, want, atol=1e-6), f"lens {lens}: expected\n{want}\ngot\n{out}"


def test_does_not_modify_input():
    lens = [2, 2]
    x = np.random.default_rng(5).normal(size=(4, H))
    before = x.copy()
    selective_batch_layer(x, lens, Wqkv, Wo)
    assert np.array_equal(x, before), "the input array must not be modified"
