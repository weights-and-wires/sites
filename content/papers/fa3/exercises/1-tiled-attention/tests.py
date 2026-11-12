import numpy as np
from submission import naive_attention, tiled_attention


def make(seed, Br=6, N=37, d=8, dv=5, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(Br, d)) * scale, rng.normal(size=(N, d)) * scale,
            rng.normal(size=(N, dv)))


def test_matches_naive_for_many_block_sizes():
    Q, K, V = make(0)
    want = naive_attention(Q, K, V, 0.3)
    for Bc in (1, 3, 8, 16, 37, 64):
        O, _ = tiled_attention(Q, K, V, Bc, 0.3)
        assert O.shape == want.shape, f"Bc={Bc}: expected shape {want.shape}, got {O.shape}"
        err = np.abs(O - want).max()
        assert err < 1e-10, f"Bc={Bc}: expected max error below 1e-10, got {err:.2e}"


def test_logsumexp_output():
    Q, K, V = make(1)
    S = 0.5 * Q @ K.T
    want = S.max(axis=1) + np.log(np.exp(S - S.max(axis=1, keepdims=True)).sum(axis=1))
    _, L = tiled_attention(Q, K, V, 5, 0.5)
    assert L.shape == want.shape, f"expected shape {want.shape}, got {L.shape}"
    assert np.allclose(L, want, atol=1e-10), f"expected L = {want}, got {L}"


def test_page_example():
    Q = np.array([[1.0]])
    K = np.array([[2.0], [0.0], [3.0], [1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    O, L = tiled_attention(Q, K, V, 2)
    assert np.allclose(O, [[1.0551, 0.6760]], atol=1e-4), f"expected O = [[1.0551, 0.6760]], got {O}"
    assert abs(L[0] - 3.4402) < 1e-4, f"expected L = 3.4402, got {L[0]}"


def test_large_scores_do_not_overflow():
    Q, K, V = make(2, scale=12.0)
    O, L = tiled_attention(Q, K, V, 4)
    assert np.all(np.isfinite(O)) and np.all(np.isfinite(L)), f"expected finite output, got O={O}, L={L}"
    want = naive_attention(Q, K, V)
    assert np.allclose(O, want, atol=1e-9), f"expected max error below 1e-9, got {np.abs(O - want).max():.2e}"


def test_running_max_grows_across_blocks():
    # The largest score sits in the last block, so every earlier block must be rescaled.
    Q = np.ones((2, 1))
    K = np.array([[0.0], [1.0], [2.0], [3.0], [8.0], [9.0]])
    V = np.arange(12, dtype=float).reshape(6, 2)
    O, _ = tiled_attention(Q, K, V, 2)
    want = naive_attention(Q, K, V)
    assert np.allclose(O, want, atol=1e-10), f"expected {want}, got {O}"


def test_scale_tau_is_applied():
    Q, K, V = make(3)
    a, _ = tiled_attention(Q, K, V, 7, 1.0)
    b, _ = tiled_attention(Q, K, V, 7, 0.1)
    want = naive_attention(Q, K, V, 0.1)
    assert np.allclose(b, want, atol=1e-10), f"expected tau=0.1 output {want}, got {b}"
    assert not np.allclose(a, b), "expected tau=1.0 and tau=0.1 to give different outputs"


def test_inputs_are_not_modified():
    Q, K, V = make(4)
    copies = [a.copy() for a in (Q, K, V)]
    tiled_attention(Q, K, V, 5)
    for name, before, after in zip("QKV", copies, (Q, K, V)):
        assert np.array_equal(before, after), f"{name} was modified"
