import numpy as np
from submission import mla_naive, mla_absorbed


def make(T=6, d=8, n_h=2, d_h=4, d_c=3, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(T, d)),
        rng.normal(size=(n_h * d_h, d)),
        rng.normal(size=(d_c, d)),
        rng.normal(size=(n_h * d_h, d_c)),
        rng.normal(size=(n_h * d_h, d_c)),
        n_h,
        d_h,
    )


def reference(H, WQ, WDKV, WUK, WUV, n_h, d_h):
    """Plain multi-head attention with full-rank keys W_K = W_UK W_DKV."""
    T = H.shape[0]
    WK, WV = WUK @ WDKV, WUV @ WDKV
    Q, K, V = (H @ WQ.T).reshape(T, n_h, d_h), (H @ WK.T).reshape(T, n_h, d_h), (H @ WV.T).reshape(T, n_h, d_h)
    out = np.zeros((T, n_h, d_h))
    for i in range(n_h):
        s = Q[:, i] @ K[:, i].T / np.sqrt(d_h)
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, -np.inf)
        w = np.exp(s - s.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        out[:, i] = w @ V[:, i]
    return out.reshape(T, n_h * d_h)


def test_shapes():
    args = make()
    for f in (mla_naive, mla_absorbed):
        got = f(*args)
        assert got.shape == (6, 8), f"{f.__name__}: expected shape (6, 8), got {got.shape}"


def test_naive_matches_full_rank_mha():
    args = make(seed=1)
    want, got = reference(*args), mla_naive(*args)
    assert np.allclose(got, want, atol=1e-8), f"expected {want[0]}, got {got[0]}"


def test_absorbed_matches_naive():
    for seed in range(3):
        args = make(seed=seed)
        want, got = mla_naive(*args), mla_absorbed(*args)
        err = np.abs(want - got).max()
        assert np.allclose(got, want, atol=1e-8), f"seed {seed}: max error {err}; expected {want[1]}, got {got[1]}"


def test_absorbed_with_unequal_widths():
    args = make(T=5, d=6, n_h=3, d_h=2, d_c=5, seed=4)
    want, got = mla_naive(*args), mla_absorbed(*args)
    assert np.allclose(got, want, atol=1e-8), f"expected {want[2]}, got {got[2]}"


def test_causal():
    H, *rest = make(seed=5)
    H2 = H.copy()
    H2[-1] += 10.0
    for f in (mla_naive, mla_absorbed):
        a, b = f(H, *rest), f(H2, *rest)
        assert np.allclose(a[:-1], b[:-1]), f"{f.__name__}: changing the last token changed earlier outputs"


def test_first_token_copies_its_value():
    H, WQ, WDKV, WUK, WUV, n_h, d_h = make(seed=6)
    want = (H[0] @ WDKV.T) @ WUV.T
    for f in (mla_naive, mla_absorbed):
        got = f(H, WQ, WDKV, WUK, WUV, n_h, d_h)[0]
        assert np.allclose(got, want), f"{f.__name__}: row 0 must equal v_0 = {want}, got {got}"


def test_scale_is_sqrt_d_h():
    args = make(seed=7, d_c=6, d_h=2)
    want = reference(*args)
    got = mla_absorbed(*args)
    assert np.allclose(got, want, atol=1e-8), f"scale must use d_h, not d_c; expected {want[3]}, got {got[3]}"
