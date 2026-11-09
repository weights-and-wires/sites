import numpy as np
from submission import flash_attention2_forward


def naive(Q, K, V, tau=1.0):
    S = tau * (Q @ K.T)
    m = S.max(axis=1)
    E = np.exp(S - m[:, None])
    l = E.sum(axis=1)
    return (E / l[:, None]) @ V, m + np.log(l)


def make(N=11, M=13, d=4, dv=5, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return scale * rng.normal(size=(N, d)), rng.normal(size=(M, d)), rng.normal(size=(M, dv))


def test_matches_naive_even_blocks():
    Q, K, V = make(N=8, M=8, dv=4)
    got, _ = flash_attention2_forward(Q, K, V, 4, 4)
    want, _ = naive(Q, K, V)
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want, atol=1e-10), f"max error {np.abs(got - want).max():.2e}\nexpected\n{want}\ngot\n{got}"


def test_matches_naive_uneven_blocks():
    Q, K, V = make()
    got, L = flash_attention2_forward(Q, K, V, 4, 5)
    want, wL = naive(Q, K, V)
    assert np.allclose(got, want, atol=1e-10), f"11 queries in blocks of 4 and 13 keys in blocks of 5 leave short last blocks; max error {np.abs(got - want).max():.2e}"
    assert np.allclose(L, wL, atol=1e-10), f"L expected {wL}, got {L}"


def test_block_sizes_do_not_change_the_answer():
    Q, K, V = make()
    want, wL = naive(Q, K, V)
    for Br, Bc in [(1, 1), (3, 2), (11, 13), (50, 50), (2, 13), (11, 1)]:
        got, L = flash_attention2_forward(Q, K, V, Br, Bc)
        assert np.allclose(got, want, atol=1e-10), f"Br={Br}, Bc={Bc}: max error {np.abs(got - want).max():.2e}"
        assert np.allclose(L, wL, atol=1e-10), f"Br={Br}, Bc={Bc}: L expected {wL}, got {L}"


def test_logsumexp_is_stored_not_m_and_l():
    # Row of scores [1, 2, 3, 4]: L = 4 + log(1 + e^-1 + e^-2 + e^-3).
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    V = np.eye(4)[:, :2]
    _, L = flash_attention2_forward(Q, K, V, 1, 2)
    want = np.log(np.exp([1.0, 2.0, 3.0, 4.0]).sum())
    assert np.allclose(L, [want]), f"expected L = log(sum(exp(scores))) = {want:.6f}, got {L}"


def test_increasing_and_decreasing_scores():
    # The running maximum changes at every block, so the unscaled output
    # must be rescaled each time and divided once at the end.
    Q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [5.0, -1.0]])
    want, wL = naive(Q, K, V)
    got, L = flash_attention2_forward(Q, K, V, 1, 1)
    assert np.allclose(got, want, atol=1e-10), f"increasing scores: expected {want}, got {got}"
    assert np.allclose(L, wL), f"increasing scores: L expected {wL}, got {L}"
    got, L = flash_attention2_forward(Q, K[::-1].copy(), V[::-1].copy(), 1, 1)
    assert np.allclose(got, want, atol=1e-10), f"decreasing scores: expected {want}, got {got}"
    assert np.allclose(L, wL), f"decreasing scores: L expected {wL}, got {L}"


def test_scale_tau():
    Q, K, V = make()
    got, L = flash_attention2_forward(Q, K, V, 4, 4, tau=0.5)
    want, wL = naive(Q, K, V, tau=0.5)
    assert np.allclose(got, want, atol=1e-10), f"tau=0.5: max error {np.abs(got - want).max():.2e}"
    assert np.allclose(L, wL), f"tau=0.5: L expected {wL}, got {L}"


def test_large_scores_stay_finite():
    Q, K, V = make(scale=400.0)
    got, L = flash_attention2_forward(Q, K, V, 4, 4)
    want, wL = naive(Q, K, V)
    assert np.all(np.isfinite(got)), f"output has NaN or inf: {got}"
    assert np.all(np.isfinite(L)), f"L has NaN or inf: {L}"
    assert np.allclose(got, want, atol=1e-8), f"max error {np.abs(got - want).max():.2e}"
    assert np.allclose(L, wL, atol=1e-6), f"L expected {wL}, got {L}"


def test_never_builds_the_full_matrix():
    Q, K, V = make(N=12, M=12)
    sizes = []
    real_exp = np.exp

    def spy(x, *a, **k):
        sizes.append(np.size(x))
        return real_exp(x, *a, **k)

    np.exp = spy
    try:
        got, _ = flash_attention2_forward(Q, K, V, 3, 4)
    finally:
        np.exp = real_exp
    assert sizes, "expected the function to call np.exp"
    assert max(sizes) <= 3 * 4, f"expected every np.exp input to have at most Br*Bc = 12 elements, saw {max(sizes)}; the 12 x 12 score matrix has 144"
    want, _ = naive(Q, K, V)
    assert np.allclose(got, want, atol=1e-10), "the answer must still match naive attention"
