import numpy as np
from submission import flash_attention_forward


def naive(Q, K, V, tau=1.0):
    S = tau * (Q @ K.T)
    m = S.max(axis=1)
    E = np.exp(S - m[:, None])
    l = E.sum(axis=1)
    return (E / l[:, None]) @ V, m, l


def make(N=11, M=13, d=4, dv=5, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return scale * rng.normal(size=(N, d)), rng.normal(size=(M, d)), rng.normal(size=(M, dv))


def test_matches_naive_even_blocks():
    Q, K, V = make(N=8, M=8, dv=4)
    got, _, _ = flash_attention_forward(Q, K, V, 4, 4)
    want, _, _ = naive(Q, K, V)
    assert np.allclose(got, want, atol=1e-10), f"max error {np.abs(got - want).max():.2e}\nexpected\n{want}\ngot\n{got}"


def test_matches_naive_uneven_blocks():
    Q, K, V = make()
    got, _, _ = flash_attention_forward(Q, K, V, 4, 5)
    want, _, _ = naive(Q, K, V)
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want, atol=1e-10), f"13 keys in blocks of 5 leave a last block of 3; max error {np.abs(got - want).max():.2e}"


def test_block_sizes_do_not_change_the_answer():
    Q, K, V = make()
    want, _, _ = naive(Q, K, V)
    for Br, Bc in [(1, 1), (3, 2), (11, 13), (50, 50), (2, 13), (11, 1)]:
        got, _, _ = flash_attention_forward(Q, K, V, Br, Bc)
        assert np.allclose(got, want, atol=1e-10), f"Br={Br}, Bc={Bc}: max error {np.abs(got - want).max():.2e}"


def test_statistics_m_and_l():
    Q, K, V = make()
    _, m, l = flash_attention_forward(Q, K, V, 3, 4)
    _, wm, wl = naive(Q, K, V)
    assert np.allclose(m, wm), f"m expected {wm}, got {m}"
    assert np.allclose(l, wl), f"l expected {wl}, got {l}"


def test_later_block_has_the_larger_maximum():
    # Scores 1, 2, 3, 4 arrive one block at a time, so the running maximum
    # changes at every step and the old output must be rescaled each time.
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [5.0, -1.0]])
    got, m, l = flash_attention_forward(Q, K, V, 1, 1)
    want, wm, wl = naive(Q, K, V)
    assert np.allclose(got, want, atol=1e-10), f"increasing scores: expected {want}, got {got}"
    assert np.allclose(l, wl), f"increasing scores: l expected {wl}, got {l}"
    got, _, l = flash_attention_forward(Q, K[::-1].copy(), V[::-1].copy(), 1, 1)
    assert np.allclose(got, want, atol=1e-10), f"decreasing scores: expected {want}, got {got}"
    assert np.allclose(l, wl), f"decreasing scores: l expected {wl}, got {l}"


def test_scale_tau():
    Q, K, V = make()
    got, m, l = flash_attention_forward(Q, K, V, 4, 4, tau=0.5)
    want, wm, wl = naive(Q, K, V, tau=0.5)
    assert np.allclose(got, want, atol=1e-10), f"tau=0.5: max error {np.abs(got - want).max():.2e}"
    assert np.allclose(m, wm), f"tau=0.5: m expected {wm}, got {m}"


def test_large_scores_stay_finite():
    Q, K, V = make(scale=400.0)
    got, m, l = flash_attention_forward(Q, K, V, 4, 4)
    want, _, _ = naive(Q, K, V)
    assert np.all(np.isfinite(got)), f"output has NaN or inf: {got}"
    assert np.allclose(got, want, atol=1e-8), f"max error {np.abs(got - want).max():.2e}"
    assert np.all(np.isfinite(l)) and np.all(l >= 1 - 1e-12), f"l must be at least 1 because the maximum term is e^0; got {l}"


def test_never_builds_the_full_matrix():
    Q, K, V = make(N=12, M=12)
    sizes = []
    real_exp = np.exp

    def spy(x, *a, **k):
        sizes.append(np.size(x))
        return real_exp(x, *a, **k)

    np.exp = spy
    try:
        got, _, _ = flash_attention_forward(Q, K, V, 3, 4)
    finally:
        np.exp = real_exp
    assert sizes, "expected the function to call np.exp"
    assert max(sizes) <= 3 * 4, f"expected every np.exp input to have at most Br*Bc = 12 elements, saw {max(sizes)}; the 12 x 12 score matrix has 144"
    want, _, _ = naive(Q, K, V)
    assert np.allclose(got, want, atol=1e-10), "the answer must still match naive attention"
