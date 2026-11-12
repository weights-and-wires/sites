import numpy as np
from submission import dsa_attention


def make(L=7, H=3, d=4, d_v=5, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(L, H, d))
    K = rng.normal(size=(L, d))
    V = rng.normal(size=(L, d_v))
    I = np.where(np.tril(np.ones((L, L), dtype=bool)), rng.normal(size=(L, L)), -np.inf)
    return Q, K, V, I


def dense_reference(Q, K, V):
    L, H, d = Q.shape
    out = np.zeros((L, H, V.shape[1]))
    for t in range(L):
        for h in range(H):
            s = K[: t + 1] @ Q[t, h] / np.sqrt(d)
            e = np.exp(s - s.max())
            out[t, h] = (e / e.sum()) @ V[: t + 1]
    return out


def test_shape():
    Q, K, V, I = make()
    got = dsa_attention(Q, K, V, I, 3)
    assert got.shape == (7, 3, 5), f"expected shape (7, 3, 5), got {got.shape}"


def test_matches_dense_when_k_covers_every_key():
    for seed in range(3):
        Q, K, V, I = make(seed=seed)
        want = dense_reference(Q, K, V)
        got = dsa_attention(Q, K, V, I, 7)
        err = np.abs(got - want).max()
        assert np.allclose(got, want, atol=1e-9), f"seed {seed}: max error {err}; expected {want[6, 0]}, got {got[6, 0]}"


def test_matches_dense_for_short_prefixes():
    # With k = 3, rows 0 to 2 see at most 3 keys and must equal dense attention.
    Q, K, V, I = make(seed=4)
    want = dense_reference(Q, K, V)
    got = dsa_attention(Q, K, V, I, 3)
    assert np.allclose(got[:3], want[:3], atol=1e-9), f"expected {want[2, 0]}, got {got[2, 0]}"


def test_k_one_copies_the_selected_value():
    Q, K, V, I = make(seed=5)
    got = dsa_attention(Q, K, V, I, 1)
    best = np.argmax(I, axis=1)
    for t in range(7):
        for h in range(3):
            assert np.allclose(got[t, h], V[best[t]]), f"t={t}, h={h}: expected V[{best[t]}] = {V[best[t]]}, got {got[t, h]}"


def test_unselected_keys_have_no_effect():
    Q, K, V, I = make(L=8, seed=6)
    k, t = 3, 7
    chosen = set(np.argsort(-I[t])[:k].tolist())
    other = next(s for s in range(8) if s not in chosen)
    K2, V2 = K.copy(), V.copy()
    K2[other] += 50.0
    V2[other] -= 50.0
    a, b = dsa_attention(Q, K, V, I, k), dsa_attention(Q, K2, V2, I, k)
    assert np.allclose(a[t], b[t]), f"key {other} is not selected by query {t}; expected {a[t, 0]}, got {b[t, 0]}"


def test_selected_weights_sum_to_one():
    # With V all ones, every output entry is the sum of the weights, which must be 1.
    Q, K, V, I = make(seed=7)
    got = dsa_attention(Q, K, np.ones_like(V), I, 2)
    assert np.allclose(got, 1.0), f"expected all ones, got {got[6, 0]}"


def test_scale_is_sqrt_d():
    Q, K, V, I = make(L=6, H=2, d=16, seed=8)
    want = dense_reference(Q, K, V)
    got = dsa_attention(Q, K, V, I, 6)
    assert np.allclose(got, want, atol=1e-9), f"scale must be 1/sqrt(d); expected {want[5, 1]}, got {got[5, 1]}"
